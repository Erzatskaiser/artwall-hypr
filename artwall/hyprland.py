import json
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger("artwall.hyprland")

def get_current_wallpaper_path() -> Optional[Path]:
    """ Headless query to hyprpaper IPC to find the active wallpaper across any monitors. """
    try:
        cmd = ["hyprctl", "hyprpaper", "listactive"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        stdout = res.stdout.strip()
        if not stdout:
            logger.error("Hyprpaper does not seem to be running or no wallpaper active.")
            return None

        # Handle multiple monitors by isolating the first reported line
        lines = stdout.splitlines()
        first_line = lines[0].strip()
        
        if ": " in first_line:
            path_str = first_line.split(": ", 1)[1].strip()
        elif " = " in first_line:
            path_str = first_line.split(" = ", 1)[1].strip()
        else:
            path_str = first_line.split(" ", 1)[-1].strip()

        p = Path(path_str)
        if p.exists():
            logger.info(f"Automatically detected current wallpaper: {p.name}")
            return p
        else:
            logger.error(f"Hyprpaper reported path that doesn't exist: {p}")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to communicate with hyprctl: {e.stderr or e}")
        return None
    except IndexError:
        logger.error(f"Malformed output from hyprctl hyprpaper listactive: {stdout}")
        return None

def get_active_monitor_info() -> Dict[str, int]:
    """ Queries hyprctl for the currently focused monitor's physical resolution. """
    try:
        res = subprocess.run(["hyprctl", "-j", "monitors"], capture_output=True, text=True, check=True)
        monitors = json.loads(res.stdout)
        
        # Look for the monitor you are currently focused on
        for mon in monitors:
            if mon.get("focused", False):
                logger.info(f"Detected active monitor '{mon['name']}' at {mon['width']}x{mon['height']}")
                return {"width": int(mon["width"]), "height": int(mon["height"])}
                
        # Fallback to the first monitor if none report as focused
        if monitors:
            mon = monitors[0]
            return {"width": int(mon["width"]), "height": int(mon["height"])}
            
    except Exception as e:
        logger.warning(f"Failed to query monitor dimensions via JSON, falling back to 1920x1080: {e}")
    
    # Safe universal fallback
    return {"width": 1920, "height": 1080}
