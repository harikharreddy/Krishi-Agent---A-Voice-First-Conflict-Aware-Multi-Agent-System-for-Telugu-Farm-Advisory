#!/bin/bash
cd "$(dirname "$0")"
source .venv-voice/bin/activate
export OMP_NUM_THREADS=8
uvicorn backend.main:app --reload --port 8000
