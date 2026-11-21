@echo off
title Student Early Stress Detection API
cd /d "%~dp0"

if not exist ".venv" (
    echo Creando entorno virtual con Python 3.12...
    py -3.12 -m venv .venv
)

call .venv\Scripts\activate.bat

echo Verificando dependencias...
pip install -q -r requirements.txt

echo.
echo ===================================================
echo  API lista en:  http://127.0.0.1:8000
echo  Documentacion: http://127.0.0.1:8000/docs
echo ===================================================
echo.

uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause