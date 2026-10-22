"""
KINEX FastAPI Server Entry Point
Hosts RESTful API, WebSocket telemetry, and Web Dashboard.

Run with:
    python run_dashboard.py
    # or
    uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.api.websocket import ws_router
from backend.database.db import init_db, STORAGE_DIR

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "apps" / "dashboard"

# Ensure database and storage directories are initialized
init_db()

app = FastAPI(
    title="KINEX — ISL Hand Sign Recognition & Telemetry API",
    description="Edge-ready gesture recognition, YOLO26 21-keypoint kinematics, and ESP32 telemetry services.",
    version="2.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/css") or path.startswith("/js"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Register API routes and WebSockets
app.include_router(api_router)
app.include_router(ws_router)

from fastapi.responses import Response, FileResponse

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", status_code=204)

# Mount media storage for recorded videos and sign assets
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

# Serve Web Dashboard with no-cache headers for instant updates
@app.get("/", include_in_schema=False)
async def serve_dashboard_root():
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        response = FileResponse(str(index_file))
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return Response(content="Dashboard not found", status_code=404)

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

