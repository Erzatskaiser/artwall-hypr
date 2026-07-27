import json
import logging
import subprocess
from pathlib import Path
from typing import Dict
from .config import Config

logger = logging.getLogger("artwall.renderer")

class WallpaperRenderer:
    def __init__(self, config: Config):
        self.cfg = config
        self.state_file = Path.home() / ".cache" / "artwall" / "current_art.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def apply_to_hyprpaper(self, image_path: Path):
        """ Applies the un-modified, original wallpaper directly via IPC. """
        path_str = str(image_path.resolve())
        try:
            # We no longer need to preload a temporary rendered file
            subprocess.run(["hyprctl", "hyprpaper", "unload", "all"], capture_output=True)
            subprocess.run(["hyprctl", "hyprpaper", "preload", path_str], capture_output=True)
            subprocess.run(["hyprctl", "hyprpaper", "wallpaper", f",{path_str}"], check=True, capture_output=True)
            logger.info(f"Applied raw wallpaper: {path_str}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Hyprpaper IPC failed: {e.stderr.decode() if e.stderr else e}")

    def render_overlay(self, meta: Dict[str, str]):
        """ Updates the state file and restarts the GTK layer-shell overlay. """
        # 1. Write the metadata to disk for the GTK app to read
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)
            
        # 2. Kill any existing overlay process to prevent duplicates
        subprocess.run(["pkill", "-f", "artwall.overlay"], capture_output=True)
        
        # 3. Spawn the new GTK overlay as a detached background process
        cmd = ["python", "-m", "artwall.overlay", str(self.state_file)]
        subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Launched Wayland GTK Overlay.")
