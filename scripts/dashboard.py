"""Open the local dashboard.

    python scripts/dashboard.py

Serves the repository to this computer only and opens dashboard/index.html, which reads the
results straight from the project's files. Stop it with Ctrl+C.
"""

import argparse
import functools
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Serve the repository on localhost and open the dashboard in the browser."""
    parser = argparse.ArgumentParser(description="Open the local dashboard.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="only start the server")
    args = parser.parse_args()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    address = f"http://127.0.0.1:{args.port}/dashboard/"
    print(f"Dashboard at {address} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
