import time
import logging
import argparse
import sys
from pathlib import Path

from .config import Config
from .hyprland import get_current_wallpaper_path
from .intelligence import ArtIntelligence
from .renderer import WallpaperRenderer

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] daemon: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )

def main():
    parser = argparse.ArgumentParser(description="Watch for wallpaper changes and update the overlay.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between checks for a wallpaper change.",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("artwall.daemon")
    
    cfg = Config()
    intel = ArtIntelligence(cfg)
    renderer = WallpaperRenderer(cfg)

    logger.info(f"Watching for wallpaper changes every {args.interval}s.")
    
    # Store the state of the last seen wallpaper
    last_wallpaper_path = None

    try:
        while True:
            current_wall = get_current_wallpaper_path()

            # If a wallpaper is detected and it is DIFFERENT from the last one we processed
            if current_wall and current_wall != last_wallpaper_path:
                
                # Only act on wallpapers inside the configured art directory
                if cfg.is_art_wallpaper(current_wall):
                    logger.info(f"Wallpaper change detected! New image: {current_wall.name}")
                    
                    try:
                        # Cache lookup, then CLIP, then the VLM
                        meta = intel.process_wallpaper(current_wall)
                        
                        # Put the result on screen
                        if meta:
                            renderer.render_overlay(meta)
                            
                        # Remember it so the same image is not reprocessed
                        last_wallpaper_path = current_wall
                        
                    except Exception as e:
                        logger.error(f"Failed to process new wallpaper: {e}")
                else:
                    logger.debug("Wallpaper changed, but it is outside the target art directory. Ignoring.")
                    # Update state anyway so we don't spam the debug log
                    last_wallpaper_path = current_wall

            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Daemon stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
