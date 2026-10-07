"""Application server launcher for CrowdWatch Control-Room System."""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("crowdwatch.server")


def main():
    parser = argparse.ArgumentParser(description="Launch CrowdWatch Control-Room Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload on code change")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload on code change (default: True)")
    args = parser.parse_args()
    reload_enabled = args.reload and not args.no_reload

    url = f"http://{args.host}:{args.port}"
    logger.info("=" * 60)
    logger.info("  CROWDWATCH: Crowd Density & Stampede Early-Warning System")
    logger.info("  Server URL: %s", url)
    logger.info("  Running fully offline on local machine (decision support only)")
    logger.info("=" * 60)

    if not args.no_browser:
        # Schedule browser opening after server binds
        import threading
        import time

        def open_browser():
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
        reload_dirs=[str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "web")],
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
