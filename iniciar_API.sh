#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creando entorno virtual..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

echo "API lista en: http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)"
uvicorn main:app --host 127.0.0.1 --port 8000 --reload