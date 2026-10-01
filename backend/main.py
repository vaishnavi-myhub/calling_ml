import sys
from pathlib import Path


# Make `python backend/main.py` and `python main.py` resolve the backend package.
backend_directory = Path(__file__).resolve().parent
workspace_directory = backend_directory.parent
if str(workspace_directory) not in sys.path:
    sys.path.insert(0, str(workspace_directory))

import uvicorn

from backend.config import settings


if __name__ == "__main__":
    if settings.reload:
        print("WARNING: RELOAD=true -- --reload has been observed to hang or leave orphaned")
        print("server processes still listening on the port after a code edit on this")
        print("machine. If the server seems to stop responding after you save a file,")
        print("that's why -- restart manually (Ctrl+C, then rerun this) instead of relying")
        print("on auto-reload, or set RELOAD=false in backend/.env.")
    uvicorn.run(
        "backend.api.server:app",
        app_dir=str(workspace_directory),
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )