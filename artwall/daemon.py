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
    parser = argparse.ArgumentParser(description="Artwall Background Daemon")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Polling interval in seconds.")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("artwall.daemon")
    
    cfg = Config()
    intel = ArtIntelligence(cfg)
    renderer = WallpaperRenderer(cfg)

    logger.info(f"Starting Artwall Daemon (Polling every {args.interval}s)...")
    
    # Store the state of the last seen wallpaper
    last_wallpaper_path = None

    try:
        while True:
            current_wall = get_current_wallpaper_path()

            # If a wallpaper is detected and it is DIFFERENT from the last one we processed
            if current_wall and current_wall != last_wallpaper_path:
                
                # Check if it's actually in our art directory
                if cfg.wallpaper_dir.parts == current_wall.parent.parts:
                    logger.info(f"Wallpaper change detected! New image: {current_wall.name}")
                    
                    try:
                        # 1. Run the AI / Local Index Pipeline
                        meta = intel.process_wallpaper(current_wall)
                        
                        # 2. Update the GTK Overlay widget
                        if meta:
                            renderer.render_overlay(meta)
                            
                        # 3. Update our state so we don't process it again
                        last_wallpaper_path = current_wall
                        
                    except Exception as e:
                        logger.error(f"Failed to process new wallpaper: {e}")
                else:
                    logger.debug("Wallpaper changed, but it is outside the target art directory. Ignoring.")
                    # Update state anyway so we don't spam the debug log
                    last_wallpaper_path = current_wall

            # Sleep to prevent CPU thrashing
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Daemon stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()
