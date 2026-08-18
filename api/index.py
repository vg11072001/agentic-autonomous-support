

import sys
import os

# make the project root importable so `backend.*` imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent.api_server import app  # noqa: E402  (path fix must come first)
from fastapi.staticfiles import StaticFiles  # noqa: E402

# serve frontend/ as a static catch-all — mounts AFTER all API routes so
# /api/* and /simulations/* are never shadowed
_frontend = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")

