#!/bin/bash
cd "$(dirname "$0")"
source .venv-voice/bin/activate
PYTHONPATH=$(pwd) streamlit run ui/app.py
