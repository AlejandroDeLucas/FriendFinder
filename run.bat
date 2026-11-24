@echo off
setlocal
set ROOT_DIR=%~dp0

echo ==^> Iniciando FriendFinder (100%% local)...

if not exist "%ROOT_DIR%backend\.venv\Scripts\python.exe" (
  echo Creando entorno virtual de Python en backend\.venv...
  python -m venv "%ROOT_DIR%backend\.venv"
  "%ROOT_DIR%backend\.venv\Scripts\pip.exe" install --upgrade pip wheel setuptools
  "%ROOT_DIR%backend\.venv\Scripts\pip.exe" install -r "%ROOT_DIR%backend\requirements.txt"
)

if not exist "%ROOT_DIR%frontend\node_modules" (
  echo Instalando dependencias del frontend...
  pushd "%ROOT_DIR%frontend"
  call npm install
  popd
)

echo ==^> Arrancando Backend FastAPI en http://127.0.0.1:8000
start "FriendFinder Backend" /D "%ROOT_DIR%backend" cmd /k ".venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000"

echo ==^> Arrancando Frontend React en http://127.0.0.1:5173
start "FriendFinder Frontend" /D "%ROOT_DIR%frontend" cmd /k "npm run dev -- --host 127.0.0.1 --port 5173"

echo.
echo Abre tu navegador en: http://127.0.0.1:5173
endlocal
