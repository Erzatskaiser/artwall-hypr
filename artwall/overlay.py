import sys
import json
from pathlib import Path
import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gtk, Gdk, GtkLayerShell, Pango

def apply_css():
    """Injects the Gruvbox Waybar styling with rounded corners."""
    css = b"""
    window {
        background-color: transparent;
    }
    #artbox {
        background-color: #282828;
        color: #ebdbb2;
        font-family: 'JetBrainsMono Nerd Font Mono', monospace;
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 10px; /* Increased from 5px to 10px for smooth rounded corners */
    }
    """
    provider = Gtk.CssProvider()
    provider.load_from_data(css)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(),
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )

def load_metadata(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)
    return {"title": "Loading...", "artist": "", "year": "", "description": ""}

def build_markup(meta: dict) -> str:
    """Uses Pango markup and strips AI-generated whitespace anomalies."""
    title_line = f"<b>{meta.get('title', 'Unknown')}</b> - {meta.get('artist', 'Unknown')}"
    if meta.get("year"):
        title_line += f", {meta.get('year')}"
        
    # Clean up the description: remove empty lines and trim trailing spaces
    raw_desc = meta.get("description", "")
    clean_lines = [line.strip() for line in raw_desc.splitlines() if line.strip()]
    clean_desc = "\n".join(clean_lines)
    
    return f"{title_line}\n{clean_desc}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m artwall.overlay <path_to_state_json>")
        sys.exit(1)
        
    state_file = Path(sys.argv[1])
    meta = load_metadata(state_file)
    
    apply_css()
    
    window = Gtk.Window()
    window.connect("destroy", Gtk.main_quit)
    
    # Initialize Layer Shell
    GtkLayerShell.init_for_window(window)
    GtkLayerShell.set_layer(window, GtkLayerShell.Layer.BOTTOM)
    
    # Pin to bottom-right
    GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.BOTTOM, True)
    GtkLayerShell.set_anchor(window, GtkLayerShell.Edge.RIGHT, True)
    
    # Set Margins: 10px right and 10px bottom for perfect symmetry with Waybar
    GtkLayerShell.set_margin(window, GtkLayerShell.Edge.RIGHT, 10)
    GtkLayerShell.set_margin(window, GtkLayerShell.Edge.BOTTOM, 10)
    
    # Text Label Configuration
    label = Gtk.Label()
    label.set_name("artbox")
    label.set_markup(build_markup(meta))
    
    # Alignment: Right-justify text, and push the block to the right edge
    label.set_justify(Gtk.Justification.RIGHT)
    label.set_xalign(1.0)
    
    # Wrapping constraints: Force a specific width to prevent vertical squishing
    label.set_line_wrap(True)
    label.set_line_wrap_mode(Pango.WrapMode.WORD)
    label.set_width_chars(45)      # Firm width creates a predictable paragraph block
    label.set_max_width_chars(45)  
    
    window.add(label)
    window.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()
