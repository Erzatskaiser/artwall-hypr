import json
import logging
import subprocess
import sys
from typing import Dict

from .config import Config

logger = logging.getLogger("artwall.renderer")


class WallpaperRenderer:
    """Hands metadata to the overlay process and restarts it."""

    def __init__(self, config: Config):
        self.cfg = config
        self.state_file = config.state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def render_overlay(self, meta: Dict[str, str]):
        """Writes the state file, then replaces any running overlay with a new one."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except OSError as e:
            logger.error(f"Could not write overlay state file: {e}")
            return

        # Replace the previous overlay rather than stacking another one on top of it.
        subprocess.run(["pkill", "-f", "artwall.overlay"], capture_output=True)

        # sys.executable keeps the overlay on this interpreter, which matters in a venv.
        cmd = [sys.executable, "-m", "artwall.overlay", str(self.state_file)]
        try:
            subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Launched overlay.")
        except OSError as e:
            logger.error(f"Could not launch overlay: {e}")

    @staticmethod
    def clear_overlay():
        """Takes the overlay off the screen."""
        subprocess.run(["pkill", "-f", "artwall.overlay"], capture_output=True)
        logger.info("Overlay cleared.")
