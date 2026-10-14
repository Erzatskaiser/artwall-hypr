import sys
import logging
import argparse

from .config import Config
from .hyprland import get_current_wallpaper_path
from .intelligence import ArtIntelligence
from .renderer import WallpaperRenderer


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Identify the active Hyprland wallpaper and annotate it with an overlay."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--clear", action="store_true", help="Remove the overlay and exit."
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("artwall.main")

    if args.clear:
        WallpaperRenderer.clear_overlay()
        return

    cfg = Config()
    intel = ArtIntelligence(cfg)
    renderer = WallpaperRenderer(cfg)

    logger.info("Detecting active wallpaper...")
    current_wall = get_current_wallpaper_path()

    if not current_wall:
        logger.error("Could not detect active wallpaper.")
        sys.exit(1)

    if not cfg.is_art_wallpaper(current_wall):
        logger.warning(
            f"{current_wall.name} is not inside {cfg.wallpaper_dir}. Skipping."
        )
        sys.exit(0)

    try:
        meta = intel.process_wallpaper(current_wall)

        if not meta or not meta.get("title"):
            logger.info(f"No metadata could be generated for {current_wall.name}.")
            sys.exit(0)

        renderer.render_overlay(meta)
        logger.info("Done.")

    except Exception as e:
        logger.error(f"Failed to process wallpaper: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
