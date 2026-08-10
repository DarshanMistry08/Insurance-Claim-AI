#!/usr/bin/env python
"""
Start the FastAPI backend server.
Run from project root: python start_api.py
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        check=True,
    )
