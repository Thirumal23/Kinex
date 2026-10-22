#!/usr/bin/env python3
"""
KINEX Localhost Web Dashboard & API Server Launcher
Launches FastAPI backend service and Web Dashboard on http://localhost:8000.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="KINEX Web Dashboard & FastAPI Service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for development")
    args = parser.parse_args()

    print("=" * 64)
    print("  KINEX — Hand Sign Processing, Recognition & Telemetry")
    print("=" * 64)
    print(f"  [+] Web Dashboard:       http://localhost:{args.port}/")
    print(f"  [+] API Documentation:   http://localhost:{args.port}/docs")
    print(f"  [+] Health Endpoint:     http://localhost:{args.port}/health")
    print(f"  [+] Telemetry WebSocket: ws://localhost:{args.port}/ws/telemetry")
    print("=" * 64)
    print(f"  Starting Uvicorn server on http://{args.host}:{args.port} ...")
    print("=" * 64)

    uvicorn.run(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
