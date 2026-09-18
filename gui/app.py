"""AI 记忆总结图形用户界面 (GUI) 主程序 - 灵动胶囊自适应全比例版。

优化亮点：
1. 右侧平滑胶囊滚动条：在内容超出或滚动时，右侧呈现极简现代胶囊滚动指示条，实时展示滚动位置，支持鼠标拖拽与滚轮同步；
2. 全比例自适应放大：彻底解决窗口放大时底部留白/粉色断层问题，使所有组件与卡片随窗口缩放等比例自适应伸展并充满视窗；
3. 文案规范：“不登录”卡片说明已移除“或公开分享页”；
4. 完好保留前期全部好评功能：晨曦极光渐变背景、胶囊输入与一键粘贴、四大多彩模式卡片、两大登录卡片、平滑渐变进度条与生成安全锁定。
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

# 确保无论从项目根目录还是 gui 目录运行，均能正确解析项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
from gui.credential_store import CredentialStoreError, WindowsCredentialStore
from gui.settings_store import (
    AppSettings,
    SettingsStoreError,
    WindowsAppSettingsStore,
    default_app_settings,
)

from gui.run_logging import GenerationRunLog
from gui.service import (
    MODE_FILENAME_SUFFIXES,
    build_document_asset_directory,
    build_image_asset_directory,
    build_output_paths,
    default_output_filename,
    fetch_chat_pipeline,
    generate_output_bundle,
    normalize_markdown_filename,
    requires_authenticated_browser,
)


def _prompt_output_target(
    parent: tk.Misc,
    modes: dict[str, bool],
    settings: AppSettings,
    suggested_name: str | None = None,
) -> tuple[Path, str] | None:
    """按设置询问输出目标；配置默认目录时只询问文件名。"""
    suggested_name = suggested_name or default_output_filename(modes)
    if settings.default_results_dir is not None:
        chosen_name = simpledialog.askstring(
            "设置保存名称",
            "请输入本次结果的保存名称：",
            initialvalue=suggested_name,
            parent=parent,
        )
        if not chosen_name or not chosen_name.strip():
            return None
        return (
            Path(settings.default_results_dir).resolve(),
            normalize_markdown_filename(chosen_name),
        )

    save_file = filedialog.asksaveasfilename(
        parent=parent,
        title=(
            "请选择保存位置并设置文件名"
            if sum(bool(value) for value in modes.values()) == 1
            else "请选择保存位置并设置共同文件名（将自动添加模式后缀）"
        ),
        initialdir=str(Path(settings.runtime_data_dir) / "results" / "summary"),
        initialfile=suggested_name,
        defaultextension=".md",
        filetypes=[("Markdown 文件", "*.md")],
    )
    if not save_file:
        return None
    selected_path = Path(save_file).resolve()
    return selected_path.parent, normalize_markdown_filename(selected_path.name)


DIRECT_SUMMARY_FILE_TYPES = {".md", ".txt"}
DIRECT_SUMMARY_MAX_BYTES = 10 * 1024 * 1024


def _load_direct_summary_file(path: Path) -> list[dict[str, str]]:
    """校验并读取可直接总结的本项目对话导出文件。"""
    path = Path(path).resolve()
    if path.suffix.lower() not in DIRECT_SUMMARY_FILE_TYPES:
        raise ValueError("仅支持 UTF-8 编码的 Markdown 或文本文件（.md/.txt）。")
    if not path.is_file():
        raise ValueError("所选文件不存在或已被移动。")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("所选文件为空。")
    if size > DIRECT_SUMMARY_MAX_BYTES:
        raise ValueError("所选文件超过 10 MiB 限制。")

    from scripts.gemini_summarizer import load_exported_markdown

    try:
        return load_exported_markdown(path)
    except UnicodeDecodeError as error:
        raise ValueError("所选文件不是有效的 UTF-8 文本。") from error


def _direct_summary_output_filename(
    source_path: Path,
    modes: dict[str, bool],
) -> str:
    """按原文件名生成直接总结的默认名称。"""
    stem = Path(source_path).stem.strip() or "AI_memory"
    enabled = [
        mode for mode in ("normal", "simple", "detailed")
        if modes.get(mode)
    ]
    if len(enabled) == 1:
        stem += MODE_FILENAME_SUFFIXES[enabled[0]]
    return f"{stem}.md"


# ==================== 颜色计算与渐变工具 ====================

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    return int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def interpolate_color(color1: str, color2: str, factor: float) -> str:
    """计算两种颜色之间的插值颜色 (0.0 ~ 1.0)"""
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)
    factor = max(0.0, min(1.0, factor))
    r = r1 + (r2 - r1) * factor
    g = g1 + (g2 - g1) * factor
    b = b1 + (b2 - b1) * factor
    return rgb_to_hex(int(r), int(g), int(b))


def draw_pill(canvas: tk.Canvas, x1, y1, x2, y2, radius=None, **kwargs):
    """在 Canvas 上绘制精细平滑的胶囊/超大圆角矩形 (Pill Shape)"""
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if radius is None:
        radius = min(w, h) / 2.0
    else:
        radius = min(radius, w / 2.0, h / 2.0)

    r = radius
    points = [
        x1 + r, y1,
        x1 + r, y1,
        x2 - r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# ==================== 右侧现代胶囊滚动指示器 ====================

class CapsuleScrollbar(tk.Canvas):
    """现代极简胶囊滚动条（支持拖拽、指示与悬停动画）"""
    def __init__(
        self,
        master,
        target_canvas: tk.Canvas,
        width: int = 10,
        bg_parent: str = "#EEF2FF",
        thumb_color: str = "#94A3B8",
        thumb_hover: str = "#6366F1",
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            highlightthickness=0,
            bg=bg_parent,
            cursor="hand2",
            **kwargs
        )
        self.target_canvas = target_canvas
        self.bar_width = width
        self.bg_parent = bg_parent
        self.thumb_color = thumb_color
        self.thumb_hover = thumb_hover
        self.is_dragging = False
        self.top_fraction = 0.0
        self.bottom_fraction = 1.0

        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Enter>", lambda e: self.redraw(hover=True))
        self.bind("<Leave>", lambda e: self.redraw(hover=False))
        self.bind("<Configure>", lambda e: self.redraw())

    def set_range(self, first: str, last: str):
        try:
            self.top_fraction = float(first)
            self.bottom_fraction = float(last)
            self.redraw()
        except Exception:
            pass

    def redraw(self, hover: bool = False):
        self.delete("all")
        w = self.winfo_width() or self.bar_width
        h = self.winfo_height()
        if h <= 20:
            return

        # 如果内容完全在视窗内无需滚动，保持极简隐形
        if (self.bottom_fraction - self.top_fraction) >= 0.999:
            return

        # 1. 绘制滑道底槽
        r = w / 2.0
        draw_pill(self, 2, 4, w - 2, h - 4, radius=r - 2, fill="#E2E8F0", outline="")

        # 2. 绘制滑块胶囊
        y1 = max(4, int(h * self.top_fraction))
        y2 = min(h - 4, max(y1 + 28, int(h * self.bottom_fraction)))
        col = self.thumb_hover if (self.is_dragging or hover) else self.thumb_color
        draw_pill(self, 1, y1, w - 1, y2, radius=r - 1, fill=col, outline="")

    def _on_click(self, event):
        h = self.winfo_height()
        if h <= 0:
            return
        self.is_dragging = True
        fraction = max(0.0, min(1.0, (event.y - 14) / float(h)))
        self.target_canvas.yview_moveto(fraction)
        self.redraw(hover=True)

    def _on_drag(self, event):
        h = self.winfo_height()
        if h <= 0:
            return
        fraction = max(0.0, min(1.0, (event.y - 14) / float(h)))
        self.target_canvas.yview_moveto(fraction)
        self.redraw(hover=True)

    def _on_release(self, event):
        self.is_dragging = False
        self.redraw(hover=False)


# ==================== 渐变胶囊按钮组件 ====================

class GradientPillButton(tk.Canvas):
    """支持渐变色、悬停光泽与状态锁定的胶囊按钮"""
    def __init__(
        self,
        master,
        text: str,
        command: callable = None,
        color_start: str = "#2563EB",
        color_end: str = "#7C3AED",
        hover_start: str = "#3B82F6",
        hover_end: str = "#8B5CF6",
        disabled_bg: str = "#CBD5E1",
        disabled_fg: str = "#64748B",
        width: int = 240,
        height: int = 48,
        font=("Microsoft YaHei", 12, "bold"),
        bg_parent: str = "#EEF2FF",
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            highlightthickness=0,
            bg=bg_parent,
            cursor="hand2",
            **kwargs
        )
        self.text = text
        self.command = command
        self.color_start = color_start
        self.color_end = color_end
        self.hover_start = hover_start
        self.hover_end = hover_end
        self.disabled_bg = disabled_bg
        self.disabled_fg = disabled_fg
        self.btn_width = width
        self.btn_height = height
        self.btn_font = font
        self.bg_parent = bg_parent

        self.is_hovered = False
        self.is_enabled = True

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda e: self.redraw())
        self.redraw()

    def set_enabled(self, enabled: bool):
        self.is_enabled = enabled
        self.config(cursor="hand2" if enabled else "arrow")
        self.redraw()

    def _on_enter(self, event=None):
        if self.is_enabled:
            self.is_hovered = True
            self.redraw()

    def _on_leave(self, event=None):
        if self.is_enabled:
            self.is_hovered = False
            self.redraw()

    def _on_click(self, event=None):
        if self.is_enabled and self.command:
            self.command()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width() or self.btn_width
        h = self.winfo_height() or self.btn_height
        r = h / 2.0

        if not self.is_enabled:
            draw_pill(self, 2, 2, w - 2, h - 2, radius=r - 2, fill=self.disabled_bg, outline="#94A3B8", width=1)
            self.create_text(
                w / 2, h / 2,
                text=self.text,
                font=self.btn_font,
                fill=self.disabled_fg
            )
            return

        c1 = self.hover_start if self.is_hovered else self.color_start
        c2 = self.hover_end if self.is_hovered else self.color_end

        steps = 36
        for i in range(steps):
            factor = i / float(steps - 1)
            band_color = interpolate_color(c1, c2, factor)
            x_start = 2 + (w - 4) * (i / steps)
            x_end = 2 + (w - 4) * ((i + 1) / steps)
            self.create_rectangle(x_start, 2, x_end, h - 2, fill=band_color, outline=band_color)

        draw_pill(self, 2, 2, w - 2, h - 2, radius=r - 2, fill="", outline="#FFFFFF", width=1.5)

        self.create_text(
            w / 2, h / 2 + 1,
            text=self.text,
            font=self.btn_font,
            fill="#1E1B4B" if self.is_hovered else "#312E81"
        )
        self.create_text(
            w / 2, h / 2,
            text=self.text,
            font=self.btn_font,
            fill="#FFFFFF"
        )


# ==================== 极美胶囊卡片选择组件 ====================

class CapsuleSelectCard(tk.Canvas):
    """完全圆滑的胶囊/大圆角主题卡片（全比例自适应响应）"""
    def __init__(
        self,
        master,
        title: str,
        subtitle: str,
        badge_text: str = "",
        is_radio: bool = False,
        initial_checked: bool = False,
        on_toggle: callable = None,
        theme_color: str = "#2563EB",
        theme_bg_light: str = "#EFF6FF",
        theme_border: str = "#93C5FD",
        width: int = 175,
        height: int = 94,
        radius: int = 20,
        bg_parent: str = "#EEF2FF",
        **kwargs
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=bg_parent,
            highlightthickness=0,
            cursor="hand2",
            **kwargs
        )
        self.title = title
        self.subtitle = subtitle
        self.badge_text = badge_text
        self.is_radio = is_radio
        self.checked = initial_checked
        self.on_toggle = on_toggle
        self.theme_color = theme_color
        self.theme_bg_light = theme_bg_light
        self.theme_border = theme_border
        self.card_w = width
        self.card_h = height
        self.radius = radius
        self.bg_parent = bg_parent
        self.disabled = False
        self.is_hovered = False

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Configure>", lambda e: self.redraw())
        self.redraw()

    def set_disabled(self, disabled: bool):
        self.disabled = disabled
        if disabled:
            self.is_hovered = False
        self.config(cursor="arrow" if disabled else "hand2")
        self.redraw()

    def _on_enter(self, event=None):
        if not self.disabled:
            self.is_hovered = True
            self.redraw()

    def _on_leave(self, event=None):
        if not self.disabled:
            self.is_hovered = False
            self.redraw()

    def _on_click(self, event=None):
        if self.disabled:
            return
        if self.is_radio:
            if not self.checked:
                self.checked = True
                self.redraw()
                if self.on_toggle:
                    self.on_toggle(self)
        else:
            self.checked = not self.checked
            self.redraw()
            if self.on_toggle:
                self.on_toggle(self)

    def set_checked(self, val: bool):
        self.checked = val
        self.redraw()

    @staticmethod
    def _fit_text_size(
        text: str,
        maximum: int,
        minimum: int,
        max_width: float,
        weight: str = "normal",
    ) -> int:
        """在给定宽度内选择尽可能大的字号，防止卡片缩小时文字重叠。"""
        max_width = max(1, int(max_width))
        for size in range(maximum, minimum - 1, -1):
            font = tkfont.Font(
                family="Microsoft YaHei",
                size=size,
                weight=weight,
            )
            if font.measure(text) <= max_width:
                return size
        return minimum

    def redraw(self):
        self.delete("all")
        w = self.winfo_width() or self.card_w
        h = self.winfo_height() or self.card_h
        r = min(self.radius, h / 2.0, w / 4.0)

        if self.disabled and self.checked:
            # 锁定期间仍保留选中主题色，让用户能看清生成时采用了哪些选项。
            bg_fill = self.theme_bg_light
            border_col = self.theme_color
            border_w = 2.0
            icon_col = self.theme_color
            title_col = "#334155"
            sub_col = "#64748B"
            badge_bg = self.theme_color
            badge_fg = "#FFFFFF"
        elif self.disabled:
            bg_fill = "#F1F5F9"
            border_col = "#E2E8F0"
            border_w = 1
            icon_col = "#CBD5E1"
            title_col = "#94A3B8"
            sub_col = "#CBD5E1"
            badge_bg = "#E2E8F0"
            badge_fg = "#94A3B8"
        elif self.checked:
            bg_fill = self.theme_bg_light
            border_col = self.theme_color
            border_w = 2.0
            icon_col = self.theme_color
            title_col = "#0F172A"
            sub_col = "#334155"
            badge_bg = self.theme_color
            badge_fg = "#FFFFFF"
        else:
            bg_fill = "#FFFFFF"
            border_col = self.theme_border if self.is_hovered else "#CBD5E1"
            border_w = 1.6 if self.is_hovered else 1.0
            icon_col = "#94A3B8"
            title_col = "#334155"
            sub_col = "#64748B"
            badge_bg = "#F1F5F9"
            badge_fg = "#64748B"

        size_scale = max(
            0.68,
            min(
                1.0,
                w / max(self.card_w, 1),
                h / max(self.card_h, 1),
            ),
        )
        horizontal_pad = max(10, round(18 * size_scale))
        header_y = max(14, round(20 * size_scale))
        indicator_radius = max(5, round(7 * size_scale))
        title_x = horizontal_pad + indicator_radius + max(
            5, round(7 * size_scale)
        )
        badge_size = max(7, round(10 * size_scale))
        subtitle_size = max(7, round(10 * size_scale))

        pad = 2
        draw_pill(
            self,
            pad, pad, w - pad, h - pad,
            radius=r,
            fill=bg_fill,
            outline=border_col,
            width=border_w
        )

        ind_x, ind_y = horizontal_pad, header_y
        if self.is_radio:
            rad = indicator_radius
            self.create_oval(
                ind_x - rad, ind_y - rad, ind_x + rad, ind_y + rad,
                outline=icon_col, width=2,
                fill=self.theme_color if self.checked else bg_fill
            )
            if self.checked:
                dot_radius = max(2, round(3 * size_scale))
                self.create_oval(
                    ind_x - dot_radius,
                    ind_y - dot_radius,
                    ind_x + dot_radius,
                    ind_y + dot_radius,
                    fill="#FFFFFF", outline=""
                )
        else:
            rad = indicator_radius
            draw_pill(
                self,
                ind_x - rad, ind_y - rad, ind_x + rad, ind_y + rad,
                radius=4,
                fill=self.theme_color if self.checked else bg_fill,
                outline=icon_col,
                width=2
            )
            if self.checked:
                check_scale = max(0.72, size_scale)
                self.create_line(
                    ind_x - 3.5 * check_scale,
                    ind_y,
                    ind_x - 1 * check_scale,
                    ind_y + 3 * check_scale,
                    ind_x + 4 * check_scale,
                    ind_y - 3 * check_scale,
                    fill="#FFFFFF",
                    width=max(1.5, 2.0 * size_scale),
                    capstyle=tk.ROUND,
                )

        show_badge = bool(self.badge_text) and w >= 138 and h >= 48
        title_right = w - horizontal_pad
        if show_badge:
            badge_font = tkfont.Font(
                family="Microsoft YaHei",
                size=badge_size,
                weight="bold",
            )
            bw = badge_font.measure(self.badge_text) + max(
                10, round(12 * size_scale)
            )
            bx2 = w - horizontal_pad
            bx1 = bx2 - bw
            badge_height = max(14, round(16 * size_scale))
            by1 = header_y - badge_height / 2
            by2 = header_y + badge_height / 2
            draw_pill(self, bx1, by1, bx2, by2, radius=7, fill=badge_bg, outline="")
            self.create_text(
                (bx1 + bx2) / 2, (by1 + by2) / 2,
                text=self.badge_text,
                font=("Microsoft YaHei", badge_size, "bold"),
                fill=badge_fg
            )
            title_right = bx1 - max(5, round(7 * size_scale))

        title_size = self._fit_text_size(
            self.title,
            maximum=max(9, round(12 * size_scale)),
            minimum=8,
            max_width=title_right - title_x,
            weight="bold",
        )
        self.create_text(
            title_x,
            header_y,
            text=self.title,
            font=("Microsoft YaHei", title_size, "bold"),
            fill=title_col,
            anchor="w",
        )

        subtitle_y = header_y + max(17, round(22 * size_scale))
        if h - subtitle_y >= max(13, subtitle_size + 4):
            self.create_text(
                horizontal_pad,
                subtitle_y,
                text=self.subtitle,
                font=("Microsoft YaHei", subtitle_size),
                fill=sub_col,
                anchor="nw",
                width=max(20, w - 2 * horizontal_pad),
            )


# ==================== 渐变平滑胶囊进度条 ====================

class CapsuleProgressBar(tk.Canvas):
    """现代纯渐变平滑胶囊进度条"""
    def __init__(self, master, height=12, bg_parent="#EEF2FF", **kwargs):
        super().__init__(
            master,
            height=height,
            bg=bg_parent,
            highlightthickness=0,
            **kwargs
        )
        self.bar_height = height
        self.progress = 0.0
        self.target_progress = 0.0
        self.bind("<Configure>", lambda e: self.redraw())

    def set_progress(self, val: float):
        self.target_progress = max(self.progress, min(1.0, float(val)))
        self._animate_step()

    def _animate_step(self):
        if self.progress < self.target_progress:
            step = max(0.015, (self.target_progress - self.progress) * 0.3)
            self.progress = min(self.target_progress, self.progress + step)
            self.redraw()
            if self.progress < self.target_progress:
                self.after(16, self._animate_step)
        else:
            self.redraw()

    def reset(self):
        self.progress = 0.0
        self.target_progress = 0.0
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.bar_height
        if w <= 10:
            return

        r = h / 2.0
        draw_pill(self, 0, 0, w, h, radius=r, fill="#CBD5E1", outline="")

        fill_w = int(w * self.progress)
        if fill_w >= h:
            steps = max(10, int(fill_w / 6))
            c_start = "#3B82F6"
            c_end = "#8B5CF6"
            for i in range(steps):
                f = i / float(steps - 1)
                color = interpolate_color(c_start, c_end, f)
                x_s = fill_w * (i / steps)
                x_e = fill_w * ((i + 1) / steps)
                self.create_rectangle(x_s, 0, x_e, h, fill=color, outline=color)
            draw_pill(self, 0, 0, fill_w, h, radius=r, fill="", outline="#FFFFFF", width=0.5)
        elif fill_w > 0:
            self.create_oval(0, 0, h, h, fill="#3B82F6", outline="")


# ==================== 大圆滑胶囊输入框容器 ====================

class CapsuleEntryBox(tk.Canvas):
    """完全圆滑的胶囊输入框（自适应宽度扩展）"""
    def __init__(self, master, on_change: callable = None, bg_parent: str = "#EEF2FF", **kwargs):
        super().__init__(master, height=48, bg=bg_parent, highlightthickness=0, **kwargs)
        self.on_change = on_change
        self.bg_parent = bg_parent

        self.inner_frame = tk.Frame(self, bg="#FFFFFF")
        self.window_id = self.create_window(0, 0, window=self.inner_frame, anchor="nw")

        self.url_entry = tk.Entry(
            self.inner_frame,
            font=("Microsoft YaHei", 10),
            bg="#FFFFFF",
            fg="#0F172A",
            relief=tk.FLAT,
            insertbackground="#2563EB"
        )
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(16, 8), ipady=3)
        if self.on_change:
            self.url_entry.bind("<KeyRelease>", lambda e: self.on_change())

        self.paste_btn = tk.Button(
            self.inner_frame,
            text="📋 粘贴链接",
            font=("Microsoft YaHei", 9, "bold"),
            bg="#EFF6FF",
            fg="#2563EB",
            activebackground="#DBEAFE",
            activeforeground="#1D4ED8",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            cursor="hand2",
            command=self._paste_clipboard
        )
        self.paste_btn.pack(side=tk.RIGHT, padx=(0, 8))

        self.bind("<Configure>", self._on_resize)

    def _paste_clipboard(self):
        try:
            text = self.winfo_toplevel().clipboard_get()
            if text:
                self.url_entry.delete(0, tk.END)
                self.url_entry.insert(0, text.strip())
                if self.on_change:
                    self.on_change()
        except Exception:
            pass

    def get_text(self) -> str:
        return self.url_entry.get().strip()

    def set_locked(self, locked: bool):
        state = tk.DISABLED if locked else tk.NORMAL
        self.url_entry.config(state=state)
        self.paste_btn.config(state=state, cursor="arrow" if locked else "hand2")

    def _on_resize(self, event):
        w = event.width
        h = 48
        self.delete("pill_bg")
        draw_pill(self, 2, 2, w - 2, h - 2, radius=22, fill="#FFFFFF", outline="#93C5FD", width=1.8, tags="pill_bg")
        self.tag_lower("pill_bg")
        self.coords(self.window_id, 10, 6)
        self.inner_frame.config(width=w - 20, height=36)
        self.inner_frame.pack_propagate(False)


class HoverTooltip:
    """为小型图标按钮提供无持久状态的悬停提示。"""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._hide()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.widget.winfo_exists():
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.attributes("-topmost", True)
        x = self.widget.winfo_rootx() + self.widget.winfo_width() - 4
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tip,
            text=self.text,
            font=("Microsoft YaHei", 8),
            foreground="#FFFFFF",
            bg="#1E293B",
            padx=8,
            pady=4,
            relief=tk.FLAT,
        ).pack()
        self._tip = tip

    def _hide(self, _event=None):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


# ==================== 主窗口与交互逻辑 ====================

class AIMemoryGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI 记忆总结协同管理工具")
        # 紧凑初始尺寸，支持自由等比例缩放
        self.root.geometry("880x660")
        self.root.minsize(780, 520)
        self.root.configure(bg="#E0E7FF")

        self.is_running = False
        self.selected_summary_file: Path | None = None
        self.credential_store = WindowsCredentialStore()
        self.settings_store = WindowsAppSettingsStore()
        try:
            self.app_settings = self.settings_store.load()
            self._settings_load_error = None
        except SettingsStoreError as error:
            self.app_settings = default_app_settings()
            self._settings_load_error = str(error)
        self._settings_dialog = None
        # 兼容缺少 API KEY 时复用已有窗口的旧属性名。
        self._api_key_dialog = None
        self._build_ui()
        self._configure_file_drop()
        self._bind_mousewheel()
        self._update_generate_button_state()

    def _build_ui(self):
        # 整体主容器
        root_container = tk.Frame(self.root, bg="#E0E7FF")
        root_container.pack(fill=tk.BOTH, expand=True)

        # 1. 主可滚动渐变画布
        self.bg_canvas = tk.Canvas(root_container, highlightthickness=0, bg="#E0E7FF")
        self.bg_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 2. 右侧现代胶囊滚动指示条
        self.custom_scrollbar = CapsuleScrollbar(
            root_container,
            target_canvas=self.bg_canvas,
            width=10,
            bg_parent="#E0E7FF"
        )
        self.custom_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bg_canvas.config(yscrollcommand=self.custom_scrollbar.set_range)

        # 3. 挂载主内容框架（全比例等比拉伸）
        self.main_content = tk.Frame(self.bg_canvas, bg="#E0E7FF")
        self.canvas_window_id = self.bg_canvas.create_window(0, 0, window=self.main_content, anchor="nw")

        self.bg_canvas.bind("<Configure>", self._on_canvas_configure)
        self.main_content.bind("<Configure>", self._on_content_configure)

        # 主卡片白底胶囊外框（fill=BOTH, expand=True 全自适应拉伸）
        self.sheet_frame = tk.Frame(self.main_content, bg="#EEF2FF", padx=28, pady=18)
        self.sheet_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        # 头部主标题区
        header_frame = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.api_key_button = tk.Button(
            header_frame,
            text="⚙",
            font=("Segoe UI Symbol", 15),
            foreground="#475569",
            bg="#EEF2FF",
            activeforeground="#2563EB",
            activebackground="#DBEAFE",
            relief=tk.FLAT,
            bd=0,
            padx=7,
            pady=2,
            cursor="hand2",
            command=self._show_settings,
        )
        self.api_key_button.place(relx=1.0, x=-2, y=0, anchor="ne")
        self._api_key_tooltip = HoverTooltip(
            self.api_key_button, "设置"
        )

        title_lbl = tk.Label(
            header_frame,
            text="AI 记忆总结",
            font=("Microsoft YaHei", 22, "bold"),
            foreground="#1E1B4B",
            bg="#EEF2FF"
        )
        title_lbl.pack(anchor="center")

        sub_chip_frame = tk.Frame(header_frame, bg="#EEF2FF")
        sub_chip_frame.pack(anchor="center", pady=(4, 0))

        sub_desc = tk.Label(
            sub_chip_frame,
            text="✨ 多平台对话提取 · 多模态深度提炼 · 极简/深度可续接记忆协同",
            font=("Microsoft YaHei", 9),
            foreground="#4338CA",
            bg="#E0E7FF",
            padx=14,
            pady=3
        )
        sub_desc.pack(anchor="center")

        # 分享链接板块（胶囊输入框）
        url_section = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        url_section.pack(fill=tk.X, pady=(0, 10))

        url_label = tk.Label(
            url_section,
            text="🔗 分享链接",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#1E293B",
            bg="#EEF2FF"
        )
        url_label.pack(anchor="w", pady=(0, 4))

        url_input_row = tk.Frame(url_section, bg="#EEF2FF")
        url_input_row.pack(fill=tk.X)

        self.capsule_entry = CapsuleEntryBox(
            url_input_row,
            on_change=self._on_url_changed,
            bg_parent="#EEF2FF"
        )
        self.capsule_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        file_action_column = tk.Frame(url_input_row, bg="#EEF2FF")
        file_action_column.pack(side=tk.RIGHT, anchor="n")
        self.file_select_button = tk.Button(
            file_action_column,
            text="📄 选取文件总结",
            font=("Microsoft YaHei", 9, "bold"),
            bg="#EEF2FF",
            fg="#4F46E5",
            activebackground="#E0E7FF",
            activeforeground="#3730A3",
            relief=tk.FLAT,
            padx=10,
            pady=7,
            cursor="hand2",
            command=self._choose_summary_file,
        )
        self.file_select_button.pack(fill=tk.X)

        self.selected_file_row = tk.Frame(file_action_column, bg="#EEF2FF")
        self.selected_file_name_var = tk.StringVar(value="")
        tk.Label(
            self.selected_file_row,
            textvariable=self.selected_file_name_var,
            font=("Microsoft YaHei", 8),
            foreground="#475569",
            bg="#EEF2FF",
            anchor="w",
            justify=tk.LEFT,
            wraplength=145,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.clear_file_button = tk.Button(
            self.selected_file_row,
            text="×",
            font=("Segoe UI", 10, "bold"),
            foreground="#64748B",
            bg="#EEF2FF",
            activeforeground="#DC2626",
            activebackground="#EEF2FF",
            relief=tk.FLAT,
            bd=0,
            padx=2,
            pady=0,
            cursor="hand2",
            command=self._clear_summary_file,
        )
        self.clear_file_button.pack(side=tk.RIGHT)

        # 导出模式（4 大彩色胶囊卡片）
        mode_section = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        mode_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        mode_header = tk.Label(
            mode_section,
            text="🏷️ 导出模式（可多选）",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#1E293B",
            bg="#EEF2FF"
        )
        mode_header.pack(anchor="w", pady=(0, 5))

        mode_grid = tk.Frame(mode_section, bg="#EEF2FF")
        mode_grid.pack(fill=tk.BOTH, expand=True)
        for i in range(4):
            mode_grid.columnconfigure(i, weight=1, uniform="mode_col")
        mode_grid.rowconfigure(0, weight=1)

        # 模式 1：raw 原始对话（暖金琥珀）
        self.card_raw = CapsuleSelectCard(
            mode_grid,
            title="仅抓取对话",
            subtitle="raw 原始问答\n不调用总结 API",
            badge_text="raw",
            is_radio=False,
            initial_checked=False,
            on_toggle=lambda c: self._on_mode_toggled(),
            theme_color="#D97706",
            theme_bg_light="#FFFBEB",
            theme_border="#FDE68A",
            bg_parent="#EEF2FF",
            height=94,
            radius=18
        )
        self.card_raw.grid(row=0, column=0, padx=5, sticky="nsew")

        # 模式 2：普通版（极光科技蓝）
        self.card_normal = CapsuleSelectCard(
            mode_grid,
            title="普通版",
            subtitle="生成分层总结\n结构化与标准摘要",
            badge_text="推荐",
            is_radio=False,
            initial_checked=True,
            on_toggle=lambda c: self._on_mode_toggled(),
            theme_color="#2563EB",
            theme_bg_light="#EFF6FF",
            theme_border="#BFDBFE",
            bg_parent="#EEF2FF",
            height=94,
            radius=18
        )
        self.card_normal.grid(row=0, column=1, padx=5, sticky="nsew")

        # 模式 3：极简版（清爽薄荷青）
        self.card_simple = CapsuleSelectCard(
            mode_grid,
            title="极简版",
            subtitle="单段落高保真总览\n去冗余极短提炼",
            badge_text="极短",
            is_radio=False,
            initial_checked=False,
            on_toggle=lambda c: self._on_mode_toggled(),
            theme_color="#0D9488",
            theme_bg_light="#F0FDFA",
            theme_border="#99F6E4",
            bg_parent="#EEF2FF",
            height=94,
            radius=18
        )
        self.card_simple.grid(row=0, column=2, padx=5, sticky="nsew")

        # 模式 4：细节版（梦幻暮光紫）
        self.card_detailed = CapsuleSelectCard(
            mode_grid,
            title="细节版",
            subtitle="总结包含细节记忆\n附带高价值要点",
            badge_text="精细",
            is_radio=False,
            initial_checked=False,
            on_toggle=lambda c: self._on_mode_toggled(),
            theme_color="#7C3AED",
            theme_bg_light="#FAF5FF",
            theme_border="#DDD6FE",
            bg_parent="#EEF2FF",
            height=94,
            radius=18
        )
        self.card_detailed.grid(row=0, column=3, padx=5, sticky="nsew")

        # 登录选项（2 大圆角胶囊卡片）
        auth_section = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        auth_section.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        auth_header = tk.Label(
            auth_section,
            text="🔐 登录选项（单选）",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#1E293B",
            bg="#EEF2FF"
        )
        auth_header.pack(anchor="w", pady=(0, 5))

        auth_grid = tk.Frame(auth_section, bg="#EEF2FF")
        auth_grid.pack(fill=tk.BOTH, expand=True)
        auth_grid.columnconfigure(0, weight=1, uniform="auth_col")
        auth_grid.columnconfigure(1, weight=1, uniform="auth_col")
        auth_grid.rowconfigure(0, weight=1)

        # 不登录（已去掉“或公开分享页”）
        self.card_no_login = CapsuleSelectCard(
            auth_grid,
            title="不登录",
            subtitle="已登录过该 AI，则不必重复登录",
            badge_text="默认",
            is_radio=True,
            initial_checked=True,
            on_toggle=self._on_auth_toggled,
            theme_color="#EA580C",
            theme_bg_light="#FFF7ED",
            theme_border="#FED7AA",
            bg_parent="#EEF2FF",
            height=80,
            radius=18
        )
        self.card_no_login.grid(row=0, column=0, padx=5, sticky="nsew")

        # 授权登录
        self.card_need_login = CapsuleSelectCard(
            auth_grid,
            title="授权登录",
            subtitle="首次总结该 AI 建议登录，将自动弹出独立浏览器",
            badge_text="首次建议",
            is_radio=True,
            initial_checked=False,
            on_toggle=self._on_auth_toggled,
            theme_color="#0284C7",
            theme_bg_light="#F0F9FF",
            theme_border="#BAE6FD",
            bg_parent="#EEF2FF",
            height=80,
            radius=18
        )
        self.card_need_login.grid(row=0, column=1, padx=5, sticky="nsew")

        # 进度反馈区（胶囊进度条与步骤提示）
        status_box = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        status_box.pack(fill=tk.X, pady=(2, 4))

        self.status_var = tk.StringVar(value="准备就绪")
        self.status_lbl = tk.Label(
            status_box,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 9),
            foreground="#475569",
            bg="#EEF2FF",
            anchor="w"
        )
        self.status_lbl.pack(side=tk.LEFT)

        self.percent_var = tk.StringVar(value="")
        self.percent_lbl = tk.Label(
            status_box,
            textvariable=self.percent_var,
            font=("Microsoft YaHei", 9, "bold"),
            foreground="#2563EB",
            bg="#EEF2FF",
            anchor="e"
        )
        self.percent_lbl.pack(side=tk.RIGHT)

        self.progress_bar = CapsuleProgressBar(self.sheet_frame, height=12, bg_parent="#EEF2FF")
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # 开始生成按钮与完成徽章区
        action_box = tk.Frame(self.sheet_frame, bg="#EEF2FF")
        action_box.pack(fill=tk.X, pady=(0, 6))

        action_buttons = tk.Frame(action_box, bg="#EEF2FF")
        action_buttons.pack(anchor="center")

        self.btn_generate = GradientPillButton(
            action_buttons,
            text="🚀 开始生成",
            command=self._on_start_generate,
            width=220,
            height=48,
            font=("Microsoft YaHei", 12, "bold"),
            bg_parent="#EEF2FF"
        )
        self.btn_generate.pack(side=tk.LEFT, padx=8)

        self.btn_direct = GradientPillButton(
            action_buttons,
            text="📝 直接总结",
            command=self._on_direct_summary,
            color_start="#0D9488",
            color_end="#4F46E5",
            hover_start="#14B8A6",
            hover_end="#6366F1",
            width=220,
            height=48,
            font=("Microsoft YaHei", 12, "bold"),
            bg_parent="#EEF2FF",
        )
        self.btn_direct.pack(side=tk.LEFT, padx=8)

        self.done_badge = tk.Label(
            action_box,
            text="🎉 生成已全部完成！文件已成功保存。",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#15803D",
            bg="#DCFCE7",
            padx=20,
            pady=5,
            relief=tk.FLAT
        )

    def _draw_vibrant_gradient(self, w: int, h: int):
        """绘制极光晨曦渐变色"""
        self.bg_canvas.delete("gradient_bg")
        c1 = "#C7D2FE"
        c2 = "#E9D5FF"
        c3 = "#FCE7F3"
        steps = 60
        half = steps // 2
        for i in range(steps):
            if i < half:
                f = i / float(half)
                col = interpolate_color(c1, c2, f)
            else:
                f = (i - half) / float(steps - half - 1)
                col = interpolate_color(c2, c3, f)
            y_s = h * (i / steps)
            y_e = h * ((i + 1) / steps)
            self.bg_canvas.create_rectangle(0, y_s, w, y_e, fill=col, outline=col, tags="gradient_bg")
        self.bg_canvas.tag_lower("gradient_bg")

    def _on_canvas_configure(self, event):
        w = event.width
        h = event.height
        if w > 10 and h > 10:
            req_h = self.main_content.winfo_reqheight()
            total_h = max(h, req_h)
            self._draw_vibrant_gradient(w, total_h)
            # 让 main_content 高度充满整个视窗，彻底杜绝下方粉色空隙断层
            self.bg_canvas.itemconfig(self.canvas_window_id, width=w, height=total_h)

    def _on_content_configure(self, event):
        # 动态更新滚轮滚动区域
        self.bg_canvas.configure(scrollregion=self.bg_canvas.bbox("all"))

    def _bind_mousewheel(self):
        """全局绑定鼠标滚轮，低分屏或缩小窗口时自由滚动并联动右侧滚动条"""
        def _on_mousewheel(event):
            can_h = self.bg_canvas.winfo_height()
            req_h = self.main_content.winfo_reqheight()
            if req_h > can_h:
                delta = -1 * int(event.delta / 120)
                self.bg_canvas.yview_scroll(delta, "units")

        self.root.bind_all("<MouseWheel>", _on_mousewheel)

    def _show_legacy_api_key_settings(self, require_key: bool = False):
        """打开只使用 Windows 凭据管理器持久化密钥的设置窗口。"""
        existing_dialog = self._api_key_dialog
        if existing_dialog is not None and existing_dialog.winfo_exists():
            existing_dialog.deiconify()
            existing_dialog.lift()
            existing_dialog.focus_force()
            if require_key and hasattr(existing_dialog, "show_notice"):
                existing_dialog.show_notice("请先配置 API KEY")
            return

        dialog = tk.Toplevel(self.root)
        self._api_key_dialog = dialog
        dialog.title("API KEY 设置")
        dialog.geometry("540x390")
        dialog.resizable(False, False)
        dialog.configure(bg="#F8FAFC")
        dialog.transient(self.root)

        panel = tk.Frame(
            dialog,
            bg="#FFFFFF",
            padx=28,
            pady=22,
            highlightthickness=1,
            highlightbackground="#DBEAFE",
        )
        panel.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

        tk.Label(
            panel,
            text="API KEY 设置",
            font=("Microsoft YaHei", 18, "bold"),
            foreground="#1E1B4B",
            bg="#FFFFFF",
        ).pack(anchor="center")

        notice_var = tk.StringVar(value="")
        notice_slot = tk.Frame(panel, bg="#FFFFFF", height=42)
        notice_slot.pack(fill=tk.X, pady=(8, 4))
        notice_slot.pack_propagate(False)
        notice_label = tk.Label(
            notice_slot,
            textvariable=notice_var,
            font=("Microsoft YaHei", 13, "bold"),
            foreground="#B91C1C",
            bg="#FEE2E2",
            padx=10,
            pady=4,
        )
        notice_after_id = {"value": None}

        def clear_notice():
            notice_var.set("")
            notice_label.pack_forget()
            notice_after_id["value"] = None

        def show_notice(message: str, duration_ms: int = 3000):
            if notice_after_id["value"] is not None:
                dialog.after_cancel(notice_after_id["value"])
            notice_var.set(message)
            notice_label.pack(anchor="center", fill=tk.X)
            notice_after_id["value"] = dialog.after(
                duration_ms, clear_notice
            )

        dialog.show_notice = show_notice

        try:
            existing_keys = self.credential_store.load_api_keys()
            load_error = None
        except CredentialStoreError as error:
            existing_keys = {}
            load_error = str(error)

        fields = tk.Frame(panel, bg="#FFFFFF")
        fields.pack(fill=tk.X)
        fields.columnconfigure(1, weight=1)

        gemini_var = tk.StringVar(value=existing_keys.get("gemini", ""))
        silicon_var = tk.StringVar(
            value=existing_keys.get("siliconflow", "")
        )
        deepseek_var = tk.StringVar(value=existing_keys.get("deepseek", ""))

        tk.Label(
            fields,
            text="Google AI Studio：",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#334155",
            bg="#FFFFFF",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12), pady=7)
        gemini_entry = tk.Entry(
            fields,
            textvariable=gemini_var,
            show="●",
            font=("Microsoft YaHei", 10),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightcolor="#60A5FA",
            highlightbackground="#CBD5E1",
        )
        gemini_entry.grid(row=0, column=1, sticky="ew", pady=7, ipady=5)

        tk.Label(
            fields,
            text="Silicon Flow：",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#334155",
            bg="#FFFFFF",
        ).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=7)
        silicon_entry = tk.Entry(
            fields,
            textvariable=silicon_var,
            show="●",
            font=("Microsoft YaHei", 10),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightcolor="#60A5FA",
            highlightbackground="#CBD5E1",
        )
        silicon_entry.grid(row=1, column=1, sticky="ew", pady=7, ipady=5)

        tk.Label(
            fields,
            text="DeepSeek：",
            font=("Microsoft YaHei", 10, "bold"),
            foreground="#334155",
            bg="#FFFFFF",
        ).grid(row=2, column=0, sticky="w", padx=(0, 12), pady=7)
        deepseek_entry = tk.Entry(
            fields,
            textvariable=deepseek_var,
            show="●",
            font=("Microsoft YaHei", 10),
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightcolor="#60A5FA",
            highlightbackground="#CBD5E1",
        )
        deepseek_entry.grid(row=2, column=1, sticky="ew", pady=7, ipady=5)

        tk.Label(
            panel,
            text=(
                "密钥仅保存到当前 Windows 用户的凭据管理器，不写入项目文件、"
                "运行日志或输出文件。"
            ),
            font=("Microsoft YaHei", 8),
            foreground="#64748B",
            bg="#FFFFFF",
            wraplength=450,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(8, 10))

        def close_dialog():
            gemini_var.set("")
            silicon_var.set("")
            deepseek_var.set("")
            self._api_key_dialog = None
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def save_keys():
            keys = {
                "gemini": gemini_var.get().strip(),
                "siliconflow": silicon_var.get().strip(),
                "deepseek": deepseek_var.get().strip(),
            }
            if require_key and not any(keys.values()):
                show_notice("请先配置 API KEY")
                return
            try:
                self.credential_store.save_api_keys(keys)
            except CredentialStoreError as error:
                show_notice(str(error), duration_ms=5000)
                return
            close_dialog()

        actions = tk.Frame(panel, bg="#FFFFFF")
        actions.pack(fill=tk.X)
        tk.Button(
            actions,
            text="取消",
            command=close_dialog,
            font=("Microsoft YaHei", 9),
            bg="#E2E8F0",
            fg="#334155",
            relief=tk.FLAT,
            padx=18,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(8, 0))
        tk.Button(
            actions,
            text="安全保存",
            command=save_keys,
            font=("Microsoft YaHei", 9, "bold"),
            bg="#2563EB",
            fg="#FFFFFF",
            activebackground="#1D4ED8",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=18,
            pady=6,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.bind("<Return>", lambda _event: save_keys())
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - 540) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - 390) // 2
        dialog.geometry(
            f"540x390+{max(0, x)}+{max(0, y)}"
        )

        dialog.grab_set()
        gemini_entry.focus_set()
        if load_error:
            show_notice(load_error, duration_ms=5000)
        elif require_key:
            show_notice("请先配置 API KEY")

    def _show_api_key_settings(self, require_key: bool = False):
        """兼容缺少密钥时的旧调用入口，并定位到设置窗口的 API 页。"""
        self._show_settings(initial_page="api", require_key=require_key)

    def _show_settings(
        self,
        initial_page: str = "api",
        require_key: bool = False,
    ):
        """打开包含 API KEY 与本地数据位置的统一设置窗口。"""
        existing_dialog = self._settings_dialog
        if existing_dialog is not None and existing_dialog.winfo_exists():
            existing_dialog.deiconify()
            existing_dialog.lift()
            existing_dialog.focus_force()
            if hasattr(existing_dialog, "show_page"):
                existing_dialog.show_page(
                    "api" if require_key else initial_page
                )
            if require_key and hasattr(existing_dialog, "show_notice"):
                existing_dialog.show_notice("请先配置 API KEY")
            return

        dialog = tk.Toplevel(self.root)
        self._settings_dialog = dialog
        self._api_key_dialog = dialog
        dialog.title("设置")
        dialog.geometry("650x470")
        dialog.resizable(False, False)
        dialog.configure(bg="#F8FAFC")
        dialog.transient(self.root)

        panel = tk.Frame(
            dialog,
            bg="#FFFFFF",
            padx=24,
            pady=18,
            highlightthickness=1,
            highlightbackground="#DBEAFE",
        )
        panel.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        api_page = tk.Frame(notebook, bg="#FFFFFF", padx=18, pady=14)
        data_page = tk.Frame(notebook, bg="#FFFFFF", padx=18, pady=14)
        notebook.add(api_page, text="API KEY 设置")
        notebook.add(data_page, text="数据保存位置")

        try:
            existing_keys = self.credential_store.load_api_keys()
            provider_order = list(
                self.credential_store.load_provider_order()
            )
            key_load_error = None
        except CredentialStoreError as error:
            existing_keys = {}
            provider_order = ["gemini", "siliconflow", "deepseek"]
            key_load_error = str(error)

        gemini_var = tk.StringVar(value=existing_keys.get("gemini", ""))
        silicon_var = tk.StringVar(
            value=existing_keys.get("siliconflow", "")
        )
        deepseek_var = tk.StringVar(value=existing_keys.get("deepseek", ""))
        settings = self.app_settings
        runtime_var = tk.StringVar(value=str(settings.runtime_data_dir))
        results_var = tk.StringVar(
            value=str(settings.default_results_dir or "")
        )

        tk.Label(
            api_page,
            text="API KEY 设置",
            font=("Microsoft YaHei", 17, "bold"),
            foreground="#1E1B4B",
            bg="#FFFFFF",
        ).pack(anchor="center")

        notice_var = tk.StringVar(value="")
        notice_slot = tk.Frame(api_page, bg="#FFFFFF", height=40)
        notice_slot.pack(fill=tk.X, pady=(7, 3))
        notice_slot.pack_propagate(False)
        notice_label = tk.Label(
            notice_slot,
            textvariable=notice_var,
            font=("Microsoft YaHei", 13, "bold"),
            foreground="#B91C1C",
            bg="#FEE2E2",
            padx=10,
            pady=4,
        )
        notice_after_id = {"value": None}

        def clear_notice():
            notice_var.set("")
            notice_label.pack_forget()
            notice_after_id["value"] = None

        def show_notice(message: str, duration_ms: int = 3000):
            if notice_after_id["value"] is not None:
                dialog.after_cancel(notice_after_id["value"])
            notice_var.set(message)
            notice_label.pack(anchor="center", fill=tk.X)
            notice_after_id["value"] = dialog.after(
                duration_ms, clear_notice
            )

        tk.Label(
            api_page,
            text="拖动左侧手柄调整兜底顺序（上方优先，未配置自动忽略）",
            font=("Microsoft YaHei", 8),
            foreground="#64748B",
            bg="#FFFFFF",
        ).pack(fill=tk.X, pady=(0, 3))

        key_fields = tk.Frame(api_page, bg="#FFFFFF")
        key_fields.pack(fill=tk.X)
        provider_vars = {
            "gemini": gemini_var,
            "siliconflow": silicon_var,
            "deepseek": deepseek_var,
        }
        provider_labels = {
            "gemini": "Google AI Studio：",
            "siliconflow": "Silicon Flow：",
            "deepseek": "DeepSeek：",
        }
        provider_rows = {}
        provider_row_parts = {}
        provider_entries = {}
        dragged_provider = {"value": None}

        def set_provider_row_active(provider: str, active: bool):
            background = "#EFF6FF" if active else "#F8FAFC"
            row, handle, label = provider_row_parts[provider]
            row.config(
                bg=background,
                highlightbackground="#60A5FA" if active else "#E2E8F0",
            )
            handle.config(bg=background)
            label.config(bg=background)

        def render_provider_rows():
            for row in provider_rows.values():
                row.pack_forget()
            for provider in provider_order:
                provider_rows[provider].pack(fill=tk.X, pady=3)
            key_fields.update_idletasks()

        def start_provider_drag(provider: str, _event=None):
            dragged_provider["value"] = provider
            set_provider_row_active(provider, True)
            return "break"

        def move_provider_drag(event):
            provider = dragged_provider["value"]
            if provider is None:
                return "break"
            pointer_y = event.widget.winfo_pointery()
            target = min(
                range(len(provider_order)),
                key=lambda index: abs(
                    pointer_y
                    - (
                        provider_rows[provider_order[index]].winfo_rooty()
                        + provider_rows[provider_order[index]].winfo_height() / 2
                    )
                ),
            )
            current = provider_order.index(provider)
            if target != current:
                provider_order.insert(target, provider_order.pop(current))
                render_provider_rows()
            return "break"

        def end_provider_drag(_event=None):
            provider = dragged_provider["value"]
            if provider is not None:
                set_provider_row_active(provider, False)
            dragged_provider["value"] = None
            return "break"

        for provider in provider_order:
            row = tk.Frame(
                key_fields,
                bg="#F8FAFC",
                padx=5,
                pady=3,
                cursor="fleur",
                highlightthickness=1,
                highlightbackground="#E2E8F0",
            )
            row.columnconfigure(2, weight=1)
            handle = tk.Label(
                row,
                text="☰",
                font=("Microsoft YaHei", 11),
                foreground="#64748B",
                bg="#F8FAFC",
                cursor="fleur",
            )
            handle.grid(row=0, column=0, padx=(2, 8))
            label = tk.Label(
                row,
                text=provider_labels[provider],
                width=17,
                anchor="w",
                font=("Microsoft YaHei", 10, "bold"),
                foreground="#334155",
                bg="#F8FAFC",
                cursor="fleur",
            )
            label.grid(row=0, column=1, sticky="w", padx=(0, 8))
            entry = tk.Entry(
                row,
                textvariable=provider_vars[provider],
                show="●",
                font=("Microsoft YaHei", 10),
                relief=tk.SOLID,
                bd=1,
                highlightthickness=1,
                highlightcolor="#60A5FA",
                highlightbackground="#CBD5E1",
            )
            entry.grid(row=0, column=2, sticky="ew", ipady=4)
            provider_rows[provider] = row
            provider_row_parts[provider] = (row, handle, label)
            provider_entries[provider] = entry
            for widget in (row, handle, label):
                widget.bind(
                    "<ButtonPress-1>",
                    lambda event, value=provider: start_provider_drag(
                        value, event
                    ),
                )
                widget.bind("<B1-Motion>", move_provider_drag)
                widget.bind("<ButtonRelease-1>", end_provider_drag)

        render_provider_rows()
        tk.Label(
            api_page,
            text=(
                "密钥仅保存到当前 Windows 用户的凭据管理器，不写入项目文件、"
                "运行日志或输出文件。"
            ),
            font=("Microsoft YaHei", 8),
            foreground="#64748B",
            bg="#FFFFFF",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(8, 10))

        tk.Label(
            data_page,
            text="数据保存位置",
            font=("Microsoft YaHei", 17, "bold"),
            foreground="#1E1B4B",
            bg="#FFFFFF",
        ).pack(anchor="center", pady=(0, 12))
        path_fields = tk.Frame(data_page, bg="#FFFFFF")
        path_fields.pack(fill=tk.X)
        path_fields.columnconfigure(0, weight=1)

        def add_path_row(row: int, label: str, variable: tk.StringVar):
            tk.Label(
                path_fields,
                text=label,
                font=("Microsoft YaHei", 10, "bold"),
                foreground="#334155",
                bg="#FFFFFF",
            ).grid(row=row * 2, column=0, columnspan=3, sticky="w", pady=(4, 4))
            entry = tk.Entry(
                path_fields,
                textvariable=variable,
                state="readonly",
                readonlybackground="#F8FAFC",
                font=("Microsoft YaHei", 9),
                relief=tk.SOLID,
                bd=1,
            )
            entry.grid(row=row * 2 + 1, column=0, sticky="ew", ipady=5)
            return entry

        runtime_entry = add_path_row(0, "运行数据保存位置", runtime_var)
        results_entry = add_path_row(
            1,
            "结果默认保存位置（未设置时每次询问）",
            results_var,
        )

        def initial_browse_dir(value: str) -> str:
            path = Path(value).expanduser() if value.strip() else Path.home()
            if path.is_dir():
                return str(path)
            if path.parent.is_dir():
                return str(path.parent)
            return str(Path.home())

        def browse_for(variable: tk.StringVar, title: str):
            selected = filedialog.askdirectory(
                parent=dialog,
                title=title,
                initialdir=initial_browse_dir(variable.get()),
                mustexist=True,
            )
            if selected:
                variable.set(str(Path(selected).resolve()))

        tk.Button(
            path_fields,
            text="浏览…",
            command=lambda: browse_for(runtime_var, "选择运行数据保存位置"),
            font=("Microsoft YaHei", 9),
            bg="#DBEAFE",
            fg="#1D4ED8",
            relief=tk.FLAT,
            padx=12,
            cursor="hand2",
        ).grid(row=1, column=1, padx=(8, 0), sticky="ns")
        tk.Button(
            path_fields,
            text="恢复默认",
            command=lambda: runtime_var.set(
                str(default_app_settings().runtime_data_dir)
            ),
            font=("Microsoft YaHei", 9),
            bg="#E2E8F0",
            fg="#334155",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2",
        ).grid(row=1, column=2, padx=(6, 0), sticky="ns")
        tk.Button(
            path_fields,
            text="浏览…",
            command=lambda: browse_for(results_var, "选择结果默认保存位置"),
            font=("Microsoft YaHei", 9),
            bg="#DBEAFE",
            fg="#1D4ED8",
            relief=tk.FLAT,
            padx=12,
            cursor="hand2",
        ).grid(row=3, column=1, padx=(8, 0), sticky="ns")
        tk.Button(
            path_fields,
            text="清除",
            command=lambda: results_var.set(""),
            font=("Microsoft YaHei", 9),
            bg="#E2E8F0",
            fg="#334155",
            relief=tk.FLAT,
            padx=10,
            cursor="hand2",
        ).grid(row=3, column=2, padx=(6, 0), sticky="ns")

        data_notice_var = tk.StringVar(
            value=str(self._settings_load_error or "")
        )
        tk.Label(
            data_page,
            textvariable=data_notice_var,
            font=("Microsoft YaHei", 8, "bold"),
            foreground="#B91C1C",
            bg="#FFFFFF",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(10, 2))
        tk.Label(
            data_page,
            text=(
                "运行数据包含浏览器登录会话、脱敏日志、总结缓存和开发调试快照。"
                "更改从下一次任务生效；旧数据不会自动搬移，授权登录可能需要重新登录一次。\n"
                "位置设置只以目录字符串保存在当前用户注册表中，不会上传。"
            ),
            font=("Microsoft YaHei", 8),
            foreground="#64748B",
            bg="#FFFFFF",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(4, 8))

        def close_dialog():
            gemini_var.set("")
            silicon_var.set("")
            deepseek_var.set("")
            self._settings_dialog = None
            self._api_key_dialog = None
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        def save_keys():
            keys = {
                "gemini": gemini_var.get().strip(),
                "siliconflow": silicon_var.get().strip(),
                "deepseek": deepseek_var.get().strip(),
            }
            if require_key and not any(keys.values()):
                show_notice("请先配置 API KEY")
                return
            try:
                self.credential_store.save_api_keys(keys)
                self.credential_store.save_provider_order(provider_order)
            except CredentialStoreError as error:
                show_notice(str(error), duration_ms=5000)
                return
            close_dialog()

        def save_locations():
            runtime_value = runtime_var.get().strip()
            results_value = results_var.get().strip()
            try:
                self.app_settings = self.settings_store.save(AppSettings(
                    runtime_data_dir=Path(runtime_value),
                    default_results_dir=(
                        Path(results_value) if results_value else None
                    ),
                ))
                self._settings_load_error = None
            except SettingsStoreError as error:
                data_notice_var.set(str(error))
                return
            close_dialog()

        def add_actions(page, save_text: str, save_command):
            actions = tk.Frame(page, bg="#FFFFFF")
            actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
            tk.Button(
                actions,
                text="取消",
                command=close_dialog,
                font=("Microsoft YaHei", 9),
                bg="#E2E8F0",
                fg="#334155",
                relief=tk.FLAT,
                padx=18,
                pady=6,
                cursor="hand2",
            ).pack(side=tk.RIGHT, padx=(8, 0))
            tk.Button(
                actions,
                text=save_text,
                command=save_command,
                font=("Microsoft YaHei", 9, "bold"),
                bg="#2563EB",
                fg="#FFFFFF",
                activebackground="#1D4ED8",
                activeforeground="#FFFFFF",
                relief=tk.FLAT,
                padx=18,
                pady=6,
                cursor="hand2",
            ).pack(side=tk.RIGHT)

        add_actions(api_page, "安全保存", save_keys)
        add_actions(data_page, "保存位置设置", save_locations)

        def show_page(page: str):
            if page == "data":
                notebook.select(data_page)
                runtime_entry.focus_set()
            else:
                notebook.select(api_page)
                provider_entries[provider_order[0]].focus_set()

        def save_current(_event=None):
            if notebook.index(notebook.select()) == 0:
                save_keys()
            else:
                save_locations()

        dialog.show_notice = show_notice
        dialog.show_page = show_page
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.bind("<Return>", save_current)
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - 650) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - 470) // 2
        dialog.geometry(f"650x470+{max(0, x)}+{max(0, y)}")
        dialog.grab_set()
        show_page("api" if require_key else initial_page)
        if key_load_error:
            show_notice(key_load_error, duration_ms=5000)
        elif require_key:
            show_notice("请先配置 API KEY")

    def _choose_summary_file(self):
        """选择并校验一个本项目导出的对话文件。"""
        current_settings = getattr(
            self, "app_settings", default_app_settings()
        )
        selected = filedialog.askopenfilename(
            parent=getattr(self, "root", None),
            title="选择要直接总结的对话文件",
            initialdir=str(current_settings.runtime_data_dir),
            filetypes=[
                ("对话 Markdown 或文本", "*.md *.txt"),
                ("Markdown 文件", "*.md"),
                ("文本文件", "*.txt"),
            ],
        )
        if not selected:
            return
        try:
            self._select_summary_file(Path(selected))
        except Exception as error:
            messagebox.showerror("无法选择文件", str(error))

    def _select_summary_file(self, selected_path: Path):
        selected_path = Path(selected_path).resolve()
        _load_direct_summary_file(selected_path)
        self.selected_summary_file = selected_path
        self.selected_file_name_var.set(selected_path.name)
        self.selected_file_row.pack(fill=tk.X, pady=(2, 0))
        self._update_generate_button_state()

    def _configure_file_drop(self):
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<DropEnter>>", self._on_file_drag_enter)
        self.root.dnd_bind("<<DropLeave>>", self._on_file_drag_leave)
        self.root.dnd_bind("<<Drop>>", self._on_file_drop)

    def _set_file_drop_highlight(self, highlighted: bool):
        self.file_select_button.config(
            bg="#DBEAFE" if highlighted else "#EEF2FF",
            fg="#3730A3" if highlighted else "#4F46E5",
        )

    def _on_file_drag_enter(self, event):
        if not self.is_running:
            self._set_file_drop_highlight(True)
        return event.action

    def _on_file_drag_leave(self, event):
        self._set_file_drop_highlight(False)
        return event.action

    def _on_file_drop(self, event):
        self._set_file_drop_highlight(False)
        if self.is_running:
            return event.action
        try:
            paths = self.root.tk.splitlist(event.data)
            if len(paths) != 1:
                raise ValueError("请一次只拖入一个文件。")
            self._select_summary_file(Path(paths[0]))
        except Exception as error:
            messagebox.showerror("无法选择文件", str(error))
        return event.action

    def _clear_summary_file(self):
        self.selected_summary_file = None
        self.selected_file_name_var.set("")
        self.selected_file_row.pack_forget()
        self._update_generate_button_state()

    def _on_mode_toggled(self):
        self._update_generate_button_state()

    def _on_auth_toggled(self, selected_card):
        if selected_card == self.card_no_login:
            self.card_need_login.set_checked(False)
        else:
            self.card_no_login.set_checked(False)
        self._update_generate_button_state()

    def _set_inputs_locked(self, locked: bool):
        """生成过程中锁定任务输入；设置入口始终可用。"""
        self.capsule_entry.set_locked(locked)
        self.card_raw.set_disabled(locked)
        self.card_normal.set_disabled(locked)
        self.card_simple.set_disabled(locked)
        self.card_detailed.set_disabled(locked)
        self.card_no_login.set_disabled(locked)
        self.card_need_login.set_disabled(locked)
        for button_name in ("file_select_button", "clear_file_button"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.config(
                    state=tk.DISABLED if locked else tk.NORMAL,
                    cursor="arrow" if locked else "hand2",
                )
        self.api_key_button.config(
            state=tk.NORMAL,
            cursor="hand2",
        )

    def _on_url_changed(self):
        """账号内会话地址仅提示后台复用登录态，不改动用户选择。"""
        url = self.capsule_entry.get_text()
        if (
            requires_authenticated_browser(url)
            and hasattr(self, "card_need_login")
        ):
            self.status_var.set(
                "检测到账号内对话链接，将先在后台复用已保存的登录状态。"
            )
        self._update_generate_button_state()
    def _update_generate_button_state(self):
        if self.is_running:
            self.btn_generate.set_enabled(False)
            if hasattr(self, "btn_direct"):
                self.btn_direct.set_enabled(False)
            return

        has_mode = any([
            self.card_raw.checked,
            self.card_normal.checked,
            self.card_simple.checked,
            self.card_detailed.checked
        ])
        has_auth = self.card_no_login.checked or self.card_need_login.checked
        has_url = len(self.capsule_entry.get_text()) > 0

        self.btn_generate.set_enabled(bool(has_mode and has_auth and has_url))
        if hasattr(self, "btn_direct"):
            has_summary_mode = any([
                self.card_normal.checked,
                self.card_simple.checked,
                self.card_detailed.checked,
            ])
            self.btn_direct.set_enabled(bool(
                getattr(self, "selected_summary_file", None)
                and has_summary_mode
            ))

    def _show_completed_badge(self, duration_sec: int = 5):
        self.done_badge.pack(anchor="center", pady=(8, 0))
        def hide():
            time.sleep(duration_sec)
            self.root.after(0, lambda: self.done_badge.pack_forget())
        threading.Thread(target=hide, daemon=True).start()

    def _on_start_generate(self):
        url = self.capsule_entry.get_text()
        if not url:
            messagebox.showwarning("提示", "请输入有效的 AI 分享链接。")
            return

        need_login = self.card_need_login.checked
        modes = {
            "raw": self.card_raw.checked,
            "normal": self.card_normal.checked,
            "simple": self.card_simple.checked,
            "detailed": self.card_detailed.checked,
        }

        if not any(modes.values()):
            messagebox.showwarning("提示", "请至少勾选一个生成模式。")
            return

        api_keys: dict[str, str] = {}
        needs_summary = any(
            modes.get(name) for name in ("normal", "simple", "detailed")
        )
        if needs_summary:
            try:
                api_keys = self.credential_store.load_api_keys()
            except CredentialStoreError:
                self._show_api_key_settings(require_key=True)
                return
            if not api_keys:
                self._show_api_key_settings(require_key=True)
                return

        # 后台任务只接收启动瞬间的副本。生成期间仍可修改设置，但新值只会
        # 用于下一次任务，不会更换本次请求的密钥、缓存或浏览器数据目录。
        api_keys = dict(api_keys)
        current_settings = getattr(
            self, "app_settings", default_app_settings()
        )
        settings = AppSettings(
            runtime_data_dir=Path(current_settings.runtime_data_dir),
            default_results_dir=(
                Path(current_settings.default_results_dir)
                if current_settings.default_results_dir is not None
                else None
            ),
        )
        output_target = _prompt_output_target(
            getattr(self, "root", None),
            modes,
            settings,
        )
        if output_target is None:
            return
        save_dir_path, output_filename = output_target

        # 锁定界面输入并启动任务
        self.is_running = True
        self.done_badge.pack_forget()
        self._set_inputs_locked(True)
        self._update_generate_button_state()
        self.progress_bar.reset()
        self.progress_bar.set_progress(0.06)
        self.status_var.set("正在启动后台引擎...")
        self.percent_var.set("6%")

        thread = threading.Thread(
            target=self._run_generation_task,
            args=(
                url,
                need_login,
                modes,
                save_dir_path,
                output_filename,
                api_keys,
                settings,
            ),
            daemon=True
        )
        thread.start()

    def _on_direct_summary(self):
        source_path = getattr(self, "selected_summary_file", None)
        if source_path is None:
            messagebox.showwarning("提示", "请先选取要直接总结的对话文件。")
            return

        modes = {
            "raw": False,
            "normal": self.card_normal.checked,
            "simple": self.card_simple.checked,
            "detailed": self.card_detailed.checked,
        }
        if not any(modes.values()):
            messagebox.showwarning(
                "提示", "请至少勾选普通版、极简版或细节版之一。"
            )
            return

        try:
            _load_direct_summary_file(source_path)
            api_keys = self.credential_store.load_api_keys()
        except CredentialStoreError:
            self._show_api_key_settings(require_key=True)
            return
        except Exception as error:
            messagebox.showerror("无法读取文件", str(error))
            return
        if not api_keys:
            self._show_api_key_settings(require_key=True)
            return

        current_settings = getattr(
            self, "app_settings", default_app_settings()
        )
        settings = AppSettings(
            runtime_data_dir=Path(current_settings.runtime_data_dir),
            default_results_dir=(
                Path(current_settings.default_results_dir)
                if current_settings.default_results_dir is not None
                else None
            ),
        )
        output_target = _prompt_output_target(
            getattr(self, "root", None),
            modes,
            settings,
            suggested_name=_direct_summary_output_filename(source_path, modes),
        )
        if output_target is None:
            return
        save_dir_path, output_filename = output_target

        self.is_running = True
        self.done_badge.pack_forget()
        self._set_inputs_locked(True)
        self._update_generate_button_state()
        self.progress_bar.reset()
        self.progress_bar.set_progress(0.06)
        self.status_var.set("正在读取所选对话文件...")
        self.percent_var.set("6%")

        threading.Thread(
            target=self._run_generation_task,
            args=(
                "",
                False,
                modes,
                save_dir_path,
                output_filename,
                dict(api_keys),
                settings,
                Path(source_path),
            ),
            daemon=True,
        ).start()

    def _confirm_login_required(self) -> bool:
        """在打开浏览器前征求用户同意。"""
        answered = threading.Event()
        result = {"confirmed": False}

        def ask():
            result["confirmed"] = messagebox.askyesno(
                "需要登录",
                "该资源需要登录后才能下载。是否打开浏览器登录后重试？",
                parent=self.root,
            )
            answered.set()

        self.root.after(0, ask)
        answered.wait()
        return result["confirmed"]

    def _show_login_dialog(self, loop: asyncio.AbstractEventLoop, login_event: asyncio.Event):
        """弹出登录提示对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("授权登录确认")
        dialog.geometry("480x230")
        dialog.resizable(False, False)
        dialog.configure(bg="#FFFFFF")
        dialog.transient(self.root)
        dialog.grab_set()

        x = self.root.winfo_x() + (self.root.winfo_width() - 480) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 230) // 2
        dialog.geometry(f"+{x}+{y}")

        content_frame = tk.Frame(dialog, bg="#FFFFFF", padx=28, pady=22)
        content_frame.pack(fill=tk.BOTH, expand=True)

        warn_title = tk.Label(
            content_frame,
            text="⚠️ 在生成结束前请勿关闭浏览器！",
            font=("Microsoft YaHei", 12, "bold"),
            foreground="#DC2626",
            bg="#FFFFFF"
        )
        warn_title.pack(anchor="w", pady=(0, 8))

        msg_body = tk.Label(
            content_frame,
            text="系统已为您打开浏览器窗口。\n请在弹出的浏览器中登录您的 AI 账号，登录成功后点击下方按钮继续生成。",
            font=("Microsoft YaHei", 10),
            foreground="#334155",
            bg="#FFFFFF",
            justify=tk.LEFT
        )
        msg_body.pack(anchor="w", pady=(0, 16))

        def on_login_done():
            loop.call_soon_threadsafe(login_event.set)
            dialog.destroy()

        btn_done = tk.Button(
            content_frame,
            text="已登录完毕，继续生成",
            font=("Microsoft YaHei", 11, "bold"),
            bg="#16A34A",
            fg="#FFFFFF",
            activebackground="#15803D",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
            command=on_login_done
        )
        btn_done.pack(anchor="center")

    def _show_summary_topic_dialog(self, available_topics, on_done):
        """在主题综合完成后，让用户选择普通/详细版要展开的具体主题。"""
        dialog = tk.Toplevel(self.root)
        dialog.title("选择重点主题")
        dialog.resizable(False, False)
        dialog.configure(bg="#F8FAFC")
        dialog.transient(self.root)
        dialog.grab_set()

        height = min(650, 270 + 78 * min(len(available_topics), 5))
        width = 610
        x = self.root.winfo_x() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")

        content = tk.Frame(dialog, bg="#F8FAFC", padx=28, pady=22)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            content,
            text="主题已经分好，请选择需要详细展示的主题",
            font=("Microsoft YaHei", 14, "bold"),
            foreground="#1E1B4B",
            bg="#F8FAFC",
        ).pack(anchor="w")
        tk.Label(
            content,
            text=(
                "所有主题摘要都会完整保留；勾选只表示该主题更重要，"
                "并在摘要后展开它的关键记忆和相关结构化记录。"
            ),
            font=("Microsoft YaHei", 9),
            foreground="#475569",
            bg="#F8FAFC",
            wraplength=550,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(5, 14))

        choices_shell = tk.Frame(content, bg="#F8FAFC")
        choices_shell.pack(fill=tk.BOTH, expand=True)
        choices_canvas = tk.Canvas(
            choices_shell,
            bg="#F8FAFC",
            highlightthickness=0,
            height=min(390, max(90, 76 * len(available_topics))),
        )
        choices_scrollbar = ttk.Scrollbar(
            choices_shell,
            orient=tk.VERTICAL,
            command=choices_canvas.yview,
        )
        choices_canvas.configure(yscrollcommand=choices_scrollbar.set)
        choices_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        choices_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        choices = tk.Frame(choices_canvas, bg="#F8FAFC")
        choices_window = choices_canvas.create_window(
            0, 0, window=choices, anchor="nw"
        )
        choices.bind(
            "<Configure>",
            lambda _event: choices_canvas.configure(
                scrollregion=choices_canvas.bbox("all")
            ),
        )
        choices_canvas.bind(
            "<Configure>",
            lambda event: choices_canvas.itemconfigure(
                choices_window, width=event.width
            ),
        )
        dialog.bind(
            "<MouseWheel>",
            lambda event: choices_canvas.yview_scroll(
                -int(event.delta / 120), "units"
            ),
        )

        variables = {}
        for index, topic in enumerate(available_topics, start=1):
            topic_id = topic["topic_id"]
            row = tk.Frame(
                choices,
                bg="#FFFFFF",
                highlightthickness=1,
                highlightbackground="#CBD5E1",
                padx=12,
                pady=8,
            )
            row.pack(fill=tk.X, pady=4)
            variable = tk.BooleanVar(master=dialog, value=False)
            variables[topic_id] = variable
            checkbox = tk.Checkbutton(
                row,
                text=f"{index}. {topic['title']}",
                variable=variable,
                font=("Microsoft YaHei", 10, "bold"),
                foreground="#1E293B",
                bg="#FFFFFF",
                activebackground="#FFFFFF",
                selectcolor="#EEF2FF",
                cursor="hand2",
                anchor="w",
            )
            checkbox.pack(fill=tk.X)
            tk.Label(
                row,
                text=(
                    str(topic.get("summary") or "该主题暂无摘要。")[:180]
                ),
                font=("Microsoft YaHei", 8),
                foreground="#64748B",
                bg="#FFFFFF",
                anchor="w",
                justify=tk.LEFT,
                wraplength=500,
            ).pack(fill=tk.X, padx=(24, 0), pady=(1, 0))

        tk.Label(
            content,
            text=(
                "未勾选主题不会消失；媒体与附件说明始终保留，"
                "详细版的细节记忆也始终保留。"
            ),
            font=("Microsoft YaHei", 8),
            foreground="#7C3AED",
            bg="#F8FAFC",
        ).pack(anchor="w", pady=(12, 8))

        completed = False

        def finish(use_checked: bool):
            nonlocal completed
            if completed:
                return
            completed = True
            selected = tuple(
                topic["topic_id"] for topic in available_topics
                if use_checked and variables[topic["topic_id"]].get()
            )
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            on_done(selected)

        button_row = tk.Frame(content, bg="#F8FAFC")
        button_row.pack(fill=tk.X, pady=(2, 0))
        tk.Button(
            button_row,
            text="不额外展开",
            command=lambda: finish(False),
            font=("Microsoft YaHei", 9),
            bg="#E2E8F0",
            fg="#334155",
            activebackground="#CBD5E1",
            relief=tk.FLAT,
            padx=16,
            pady=7,
            cursor="hand2",
        ).pack(side=tk.LEFT)
        tk.Button(
            button_row,
            text="确认重点主题并继续",
            command=lambda: finish(True),
            font=("Microsoft YaHei", 10, "bold"),
            bg="#4F46E5",
            fg="#FFFFFF",
            activebackground="#4338CA",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            padx=18,
            pady=8,
            cursor="hand2",
        ).pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.bind("<Escape>", lambda _event: finish(False))
        dialog.bind("<Return>", lambda _event: finish(True))
        dialog.focus_force()

    def _select_summary_topics(
        self,
        result,
        update_progress,
        run_log: GenerationRunLog,
    ) -> tuple[tuple[str, ...], float]:
        """在 GUI 线程询问重点主题，供两种总结入口共用。"""
        from scripts.gemini_summarizer import available_summary_topics

        available = available_summary_topics(result)
        if not available:
            update_progress(
                0.80,
                "主题分类完成，本次没有需要单独选择的历史主题。",
            )
            return (), 0.0

        update_progress(
            0.80,
            "主题分类完成，请在弹出的窗口中勾选重要主题...",
        )
        run_log.event("topic_selection_started", topic_count=len(available))
        selection_ready = threading.Event()
        selection_holder = {"topics": ()}

        def on_selected(topics):
            selection_holder["topics"] = tuple(topics)
            selection_ready.set()

        def show_dialog():
            try:
                self._show_summary_topic_dialog(available, on_selected)
            except Exception:
                on_selected(())

        self.root.after(0, show_dialog)
        selection_started = time.perf_counter()
        selection_ready.wait()
        wait_seconds = time.perf_counter() - selection_started
        selected = selection_holder["topics"]
        run_log.event(
            "topic_selection_completed",
            selected_count=len(selected),
            wait_seconds=round(wait_seconds, 3),
        )
        update_progress(
            0.84,
            (
                f"已选择 {len(selected)} 个重点主题，正在写入结果..."
                if selected
                else "未选择重点主题，正在写入完整主题摘要..."
            ),
        )
        return selected, wait_seconds

    def _run_generation_task(
        self,
        url: str,
        need_login: bool,
        modes: dict[str, bool],
        save_dir: Path,
        output_filename: str,
        api_keys: dict[str, str],
        app_settings: AppSettings,
        source_path: Path | None = None,
    ):
        """后台异步流水线调用与平滑进度控制"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task_started = time.perf_counter()
        run_succeeded = False
        direct_summary = source_path is not None
        if direct_summary:
            source_path = Path(source_path).resolve()
            metadata = {
                "input_file_name": source_path.name,
                "input_file_fingerprint": hashlib.sha256(
                    str(source_path).encode("utf-8")
                ).hexdigest()[:12],
                "direct_summary": True,
            }
        else:
            metadata = {
                "link_host": urlparse(url).netloc.lower(),
                "link_fingerprint": hashlib.sha256(
                    url.encode("utf-8")
                ).hexdigest()[:12],
                "need_login": need_login,
            }
        metadata.update({
            "modes": [key for key, enabled in modes.items() if enabled],
            "output_filename": output_filename,
        })
        run_log = GenerationRunLog(
            app_settings.log_dir,
            metadata=metadata,
        )

        login_event = asyncio.Event()

        def update_progress(val: float, msg: str):
            run_log.event(
                "progress",
                msg,
                ui_progress=round(float(val), 3),
            )
            percent_str = f"{int(val * 100)}%"
            self.root.after(0, lambda: [
                self.progress_bar.set_progress(val),
                self.status_var.set(msg),
                self.percent_var.set(percent_str)
            ])

        try:
            fetch_started = time.perf_counter()
            source_name = None
            source_dir = None
            if direct_summary:
                update_progress(0.15, "正在读取所选对话文件...")
                messages = _load_direct_summary_file(source_path)
                fetch_seconds = time.perf_counter() - fetch_started
                fetch_active_seconds = fetch_seconds
                login_wait_seconds = 0.0
                source_name = source_path.name
                source_dir = source_path.parent
                run_log.event(
                    "file_loaded",
                    "本地对话文件读取完成",
                    elapsed_seconds=round(fetch_seconds, 3),
                    message_count=len(messages),
                )
                update_progress(
                    0.42,
                    f"成功读取 {len(messages)} 条对话消息，正在准备总结...",
                )
            else:
                update_progress(0.15, "正在加载分享页并解析动态列表...")
                output_paths = build_output_paths(
                    save_dir, modes, output_filename
                )
                asset_owner = output_paths["asset_markdown"].name
                image_output_dir = build_image_asset_directory(
                    save_dir,
                    asset_owner,
                )
                document_output_dir = build_document_asset_directory(
                    save_dir,
                    asset_owner,
                )
                fetch_res = loop.run_until_complete(
                    fetch_chat_pipeline(
                        url=url,
                        need_login=need_login,
                        login_ready_event=login_event,
                        login_required_callback=(
                            lambda: self.root.after(
                                0,
                                lambda: self._show_login_dialog(
                                    loop, login_event
                                ),
                            )
                        ),
                        login_confirmation_callback=(
                            self._confirm_login_required
                        ),
                        logger=lambda m: update_progress(0.28, m),
                        image_output_dir=image_output_dir,
                        image_reference_base=save_dir,
                        document_output_dir=document_output_dir,
                        document_reference_base=save_dir,
                        browser_profile_root=app_settings.browser_profile_dir,
                        debug_html_file=app_settings.debug_html_file,
                    )
                )
                fetch_seconds = time.perf_counter() - fetch_started
                login_wait_seconds = max(0.0, fetch_res.user_wait_seconds)
                fetch_active_seconds = max(
                    0.0, fetch_seconds - login_wait_seconds
                )
                run_log.event(
                    "fetch_completed" if not fetch_res.error else "fetch_failed",
                    fetch_res.error or "抓取完成",
                    elapsed_seconds=round(fetch_seconds, 3),
                    active_seconds=round(fetch_active_seconds, 3),
                    login_wait_seconds=round(login_wait_seconds, 3),
                    message_count=len(fetch_res.messages),
                    downloaded_images=len(fetch_res.image_map),
                )
                for warning in fetch_res.warnings:
                    run_log.event("fetch_warning", warning)
                if fetch_res.error or not fetch_res.messages:
                    err = fetch_res.error or "未能提取到有效对话内容。"
                    self.root.after(
                        0,
                        lambda: messagebox.showerror("生成失败", err),
                    )
                    return

                messages = fetch_res.messages
                login_wait_detail = (
                    f"，等待登录 {login_wait_seconds:.1f} 秒"
                    if login_wait_seconds >= 0.1
                    else ""
                )
                update_progress(
                    0.42,
                    f"成功提取 {len(messages)} 条对话交互"
                    f"（实际抓取 {fetch_active_seconds:.1f} 秒"
                    f"{login_wait_detail}），正在按要求生成文件...",
                )

            selection_wait_seconds = 0.0

            def select_summary_topics(result):
                nonlocal selection_wait_seconds
                selected, wait_seconds = self._select_summary_topics(
                    result, update_progress, run_log
                )
                selection_wait_seconds += wait_seconds
                return selected

            update_progress(0.56, "正在连接总结后端并准备生成文件...")
            generation_started = time.perf_counter()
            bundle = generate_output_bundle(
                messages=messages,
                modes=modes,
                save_dir=save_dir,
                output_filename=output_filename,
                project_dir=app_settings.runtime_data_dir,
                api_keys=api_keys,
                result_cache_dir=app_settings.summary_cache_dir,
                source_platform=(
                    "deepseek"
                    if (
                        not direct_summary
                        and urlparse(url).hostname == "chat.deepseek.com"
                    )
                    else None
                ),
                source_name=source_name,
                source_dir=source_dir,
                topic_selector=(
                    select_summary_topics
                    if modes.get("normal") or modes.get("detailed")
                    else None
                ),
                progress=lambda message: update_progress(0.72, message),
            )
            generation_seconds = (
                time.perf_counter() - generation_started - selection_wait_seconds
            )
            saved_files = [str(path.resolve()) for path in bundle.saved_files]
            processing = (
                (bundle.summary_result or {}).get("processing", {})
                if isinstance(bundle.summary_result, dict)
                else {}
            )
            for warning in processing.get("warnings", []):
                run_log.event("generation_warning", warning)
            run_log.event(
                "generation_completed",
                fetch_seconds=round(fetch_seconds, 3),
                fetch_active_seconds=round(fetch_active_seconds, 3),
                login_wait_seconds=round(login_wait_seconds, 3),
                generation_seconds=round(generation_seconds, 3),
                selection_wait_seconds=round(selection_wait_seconds, 3),
                cache_hit=processing.get("cache_hit"),
                stage_timings=processing.get("timings_seconds", {}),
                saved_files=saved_files,
            )

            # 4. 完成进度 100%
            file_list_str = "、".join(saved_files)
            total_seconds = time.perf_counter() - task_started
            program_seconds = fetch_active_seconds + generation_seconds
            wait_seconds = login_wait_seconds + selection_wait_seconds
            wait_parts = []
            if login_wait_seconds >= 0.1:
                wait_parts.append(f"登录 {login_wait_seconds:.1f}")
            if selection_wait_seconds >= 0.1:
                wait_parts.append(f"选题 {selection_wait_seconds:.1f}")
            wait_text = (
                f"；人工等待 {wait_seconds:.1f} 秒"
                f"（{'/'.join(wait_parts)}）"
                if wait_parts
                else ""
            )
            update_progress(
                1.0,
                (
                    f"{'直接总结' if direct_summary else '所有任务生成'}完成"
                    f"（程序处理 {program_seconds:.1f} 秒："
                    f"{'读取' if direct_summary else '抓取'} "
                    f"{fetch_active_seconds:.1f}/"
                    f"总结 {generation_seconds:.1f}"
                    f"{wait_text}；总计 {total_seconds:.1f} 秒）："
                    f"{file_list_str}"
                ) if file_list_str else "所有任务生成完成！",
            )
            self.root.after(0, lambda: self._show_completed_badge(5))
            run_succeeded = True

        except Exception as e:
            err_msg = "处理失败，请检查网络、额度和模型配置。"
            try:
                from scripts.gemini_summarizer import safe_error_message
                err_msg = safe_error_message(e, tuple(api_keys.values()))
            except Exception:
                pass
            run_log.event(
                "run_error",
                err_msg,
                error_type=type(e).__name__,
            )
            update_progress(0.0, f"❌ 处理发生错误: {err_msg}")
            self.root.after(0, lambda: messagebox.showerror("处理失败", f"生成失败：{err_msg}"))

        finally:
            run_log.event(
                "run_finished",
                succeeded=run_succeeded,
                total_seconds=round(time.perf_counter() - task_started, 3),
            )
            run_log.close()
            loop.close()
            self.root.after(0, self._on_task_finished)

    def _on_task_finished(self):
        self.is_running = False
        self._set_inputs_locked(False)
        self._update_generate_button_state()


def main():
    if "--package-smoke-report" in sys.argv:
        option_index = sys.argv.index("--package-smoke-report")
        try:
            report_path = Path(sys.argv[option_index + 1]).resolve()
        except IndexError:
            raise SystemExit("--package-smoke-report 后必须提供报告路径")
        from gui.package_smoke import run_package_smoke_test

        raise SystemExit(run_package_smoke_test(report_path))

    root = TkinterDnD.Tk()
    app = AIMemoryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
