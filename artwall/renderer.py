import subprocess
import logging
import textwrap
from pathlib import Path
from typing import Dict
from .config import Config
from .hyprland import get_active_monitor_info

logger = logging.getLogger("artwall.renderer")

class WallpaperRenderer:
    def __init__(self, config: Config):
        self.cfg = config

    def _assemble_text(self, meta: Dict[str, str]) -> str:
        """ Formats text into a compact block and adds padding for the background box. """
        line1 = f"{meta['title']} — {meta['artist']}"
        if meta.get('year'):
            line1 += f", {meta['year']}"
            
        # Add visual padding to the title line to mimic Waybar's 'padding: 0 10px;'[cite: 2]
        line1 = f"  {line1}  "
        
        raw_desc = meta.get('description', '')
        lines = raw_desc.splitlines()
        wrapped_lines = []
        
        for line in lines:
            clean_line = " ".join(line.split())
            if clean_line:
                # Wrap at 50 chars to keep the block compact
                wrapped = textwrap.fill(clean_line, width=50)
                # Apply the same horizontal padding spaces to the description lines
                for w_line in wrapped.splitlines():
                    wrapped_lines.append(f"  {w_line}  ")
        
        formatted_desc = "\n".join(wrapped_lines)
        return f"{line1}\n{formatted_desc}"

    def burn_multiline_text(self, image_path: Path, meta: Dict[str, str]) -> Path:
        """ Burns text with a solid Gruvbox background block dynamically scaled to the monitor. """
        formatted_text = self._assemble_text(meta)
        output_path = self.cfg.temp_rendered_path

        # Grab exact physical monitor resolution from Hyprland
        mon = get_active_monitor_info()
        w, h = mon["width"], mon["height"]

        # Calculate dynamic typography and margin values
        dynamic_font_size = max(16, int(h * self.cfg.font_size_pct))
        dynamic_spacing = max(4, int(dynamic_font_size * 0.2))
        
        offset_x = int(w * self.cfg.margin_x_pct)
        offset_y = int(h * self.cfg.margin_y_pct)
        dynamic_offset = f"+{offset_x}+{offset_y}"

        logger.debug(f"Scaling for {w}x{h}: Font={dynamic_font_size}pt, Offset={dynamic_offset}")

        # Pre-crop canvas and annotate using -undercolor for the background box
        cmd = [
            "magick", str(image_path),
            "-resize", f"{w}x{h}^",
            "-gravity", "center",
            "-extent", f"{w}x{h}",
            "-gravity", self.cfg.gravity,
            "-font", self.cfg.font,
            "-pointsize", str(dynamic_font_size),
            "-interline-spacing", str(dynamic_spacing),
            "-fill", self.cfg.fill_color,
            "-undercolor", self.cfg.box_color,           # Adds the solid Gruvbox dark background block
            "-stroke", self.cfg.stroke_color,
            "-strokewidth", str(self.cfg.stroke_width),
            "-annotate", dynamic_offset, formatted_text,
            str(output_path)
        ]

        logger.debug(f"Running ImageMagick: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    def apply_to_hyprpaper(self, image_path: Path):
        """ Reloads hyprpaper safely without crashing on cached GPU textures. """
        path_str = str(image_path.resolve())
        try:
            subprocess.run(["hyprctl", "hyprpaper", "unload", "all"], capture_output=True)
            subprocess.run(["hyprctl", "hyprpaper", "preload", path_str], capture_output=True)
            subprocess.run(["hyprctl", "hyprpaper", "wallpaper", f",{path_str}"], check=True, capture_output=True)
            logger.info(f"Successfully applied annotated wallpaper: {path_str}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Hyprpaper IPC failed: {e.stderr.decode() if e.stderr else e}")
            raise
