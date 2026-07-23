import sys
import logging
import argparse
from pathlib import Path

from artwall.config import Config
from artwall.hyprland import get_current_wallpaper_path
from artwall.intelligence import ArtIntelligence
from artwall.renderer import WallpaperRenderer

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )

def main():
    parser = argparse.ArgumentParser(description="AI-powered wallpaper metadata recovery for Hyprland.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("artwall.main")

    cfg = Config()
    intel = ArtIntelligence(cfg.ollama_url, cfg.vlm_model)
    renderer = WallpaperRenderer(cfg)

    logger.info("Detecting active wallpaper...")
    current_wall = get_current_wallpaper_path()
    
    if not current_wall:
        sys.exit(1)

    if cfg.wallpaper_dir.parts != current_wall.parent.parts:
        logger.warning(f"Active wallpaper {current_wall.name} is not inside target directory {cfg.wallpaper_dir}. Skipping.")
        sys.exit(0)

    meta = intel.identify_via_vlm(current_wall)

    if not meta or not meta.get("title"):
        logger.info(f"VLM uncertain or identification failed for {current_wall.name}. Wallpaper remains unchanged.")
        sys.exit(0)

    logger.info("Painting identified. Generating annotation...")
    try:
        temp_wall_path = renderer.burn_multiline_text(current_wall, meta)
        renderer.apply_to_hyprpaper(temp_wall_path)
        logger.info("Done.")
    except Exception as e:
        logger.error(f"Failed to process wallpaper: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
