@echo off
setlocal
set ROOT_DIR=%~dp0
if "%PORT%"=="" set PORT=8000

echo ==^> Preparando FriendFinder para acceso movil con Cloudflare Tunnel...

where cloudflared >nul 2>nul
if errorlevel 1 (
  echo.
  echo No se encontro 'cloudflared'. Instalalo con:
  echo   winget install --id Cloudflare.cloudflared
  exit /b 1
)

if not exist "%ROOT_DIR%backend\.venv\Scripts\python.exe" (
  echo ==^> Creando entorno virtual de Python...
  python -m venv "%ROOT_DIR%backend\.venv"
  "%ROOT_DIR%backend\.venv\Scripts\pip.exe" install --upgrade pip wheel setuptools
  "%ROOT_DIR%backend\.venv\Scripts\pip.exe" install -r "%ROOT_DIR%backend\requirements.txt"
)

if not exist "%ROOT_DIR%frontend\node_modules" (
  echo ==^> Instalando dependencias del frontend...
  pushd "%ROOT_DIR%frontend"
  call npm install
  popd
)

echo ==^> Compilando interfaz web...
pushd "%ROOT_DIR%frontend"
call npm run build
popd

echo ==^> Iniciando servidor en http://127.0.0.1:%PORT%...
start "FriendFinder Server" /D "%ROOT_DIR%backend" cmd /k ".venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port %PORT%"

timeout /t 3 /nobreak >nul

echo ==^> Abriendo tunel HTTPS de Cloudflare (abre la URL https://...trycloudflare.com en tu movil)...
cloudflared tunnel --url http://127.0.0.1:%PORT%
endlocal
