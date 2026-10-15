#!/usr/bin/env bash
# One-time setup for macOS / Linux. Run:  bash setup.sh
set -e
cd "$(dirname "$0")"
echo "Creating virtual environment in .venv ..."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo
echo "Setup complete! Try:"
echo "   source .venv/bin/activate"
echo "   python lessons/01_why_rag.py"
echo "   python app.py"
echo "   streamlit run web_app.py"
