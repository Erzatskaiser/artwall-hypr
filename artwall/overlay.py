"""
GTK layer-shell overlay.

Run as a standalone process so the main pipeline can replace it on every wallpaper
change without holding a GTK main loop open itself:

    python -m artwall.overlay ~/.cache/artwall/current_art.json
"""

import html
import json
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gtk, Gdk, GtkLayerShell, Pango  # noqa: E402

from .config import Config  # noqa: E402


def apply_css(cfg: Config):
    """Styles the text box from the config so it can match the rest of your desktop."""
    css = f"""
    window {{
        background-color: transparent;
    }}
    #artbox {{
        background-color: {cfg.bg_color};
        color: {cfg.fg_color};
        font-family: '{cfg.font_family}', monospace;
        font-size: {cfg.font_size}px;
        padding: 8px 12px;
        border-radius: {cfg.corner_radius}px;
    }}
    """
    provider = Gtk.CssProvider()
    provider.load_from_data(css.encode("utf-8"))
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


def load_metadata(state_file: Path) -> dict:
    """Reads the state file, falling back to placeholder text if it is missing or bad."""
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"title": "No artwork data", "artist": "", "year": "", "description": ""}
    except (json.JSONDecodeError, OSError):
        return {"title": "Unreadable state file", "artist": "", "year": "", "description": ""}


def build_markup(meta: dict) -> str:
    """
    Builds the Pango markup for the overlay.

    Values are escaped because titles and model output can contain & or <, which would
    otherwise make Pango reject the whole string and leave the overlay blank.
    """
    esc = lambda value: html.escape(str(value or "").strip())  # noqa: E731

    title = esc(meta.get("title")) or "Unknown"
    artist = esc(meta.get("artist")) or "Unknown"
    title_line = f"<b>{title}</b> - {artist}"

    year = esc(meta.get("year"))
    if year:
        title_line += f", {year}"

    raw_desc = str(meta.get("description") or "")
    clean_lines = [esc(line) for line in raw_desc.splitlines() if line.strip()]
    clean_desc = "\n".join(clean_lines)

    return f"{title_line}\n{clean_desc}" if clean_desc else title_line


def main():
    if len(sys.argv) < 2:
        print("usage: python -m artwall.overlay <path to state json>")
        sys.exit(1)

    cfg = Config()
    state_file = Path(sys.argv[1])
    meta = load_metadata(state_file)

    apply_css(cfg)

    window = Gtk.Window()
    window.connect("destroy", Gtk.main_quit)

    GtkLayerShell.init_for_window(window)
    GtkLayerShell.set_layer(window, GtkLayerShell.Layer.BOTTOM)
    GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.BOTTOM, True)
    GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.RIGHT, True)
    GtkLayerShell.set_margin(window, GtkLayerShell.Edge.RIGHT, cfg.margin_x)
    GtkLayerShell.set_margin(window, GtkLayerShell.Edge.BOTTOM, cfg.margin_y)

    label = Gtk.Label()
    label.set_name("artbox")
    label.set_markup(build_markup(meta))
    label.set_justify(Gtk.Justification.RIGHT)
    label.set_xalign(1.0)

    # A fixed width keeps the block a predictable paragraph rather than one long line.
    label.set_line_wrap(True)
    label.set_line_wrap_mode(Pango.WrapMode.WORD)
    label.set_width_chars(cfg.width_chars)
    label.set_max_width_chars(cfg.width_chars)

    window.add(label)
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
