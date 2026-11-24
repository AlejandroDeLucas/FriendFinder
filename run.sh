#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Iniciando FriendFinder (100% local)..."

if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
  echo "Creando entorno virtual de Python en backend/.venv..."
  python3 -m venv "$ROOT_DIR/backend/.venv"
  "$ROOT_DIR/backend/.venv/bin/pip" install --upgrade pip wheel setuptools
  "$ROOT_DIR/backend/.venv/bin/pip" install -r "$ROOT_DIR/backend/requirements.txt"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "Instalando dependencias del frontend..."
  (cd "$ROOT_DIR/frontend" && npm install)
fi

echo "==> Arrancando Backend FastAPI en http://127.0.0.1:8000"
(cd "$ROOT_DIR/backend" && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000) &
BACKEND_PID=$!

echo "==> Arrancando Frontend React en http://127.0.0.1:5173"
(cd "$ROOT_DIR/frontend" && npm run dev -- --host 127.0.0.1 --port 5173) &
FRONTEND_PID=$!

cleanup() {
  echo ""
  echo "Deteniendo FriendFinder..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "FriendFinder listo en: http://127.0.0.1:5173"
echo "Pulsa Ctrl+C para salir."
wait
