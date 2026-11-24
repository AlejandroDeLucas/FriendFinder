#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"

echo "==> Preparando FriendFinder para acceso móvil con Cloudflare Tunnel..."

if ! command -v cloudflared >/dev/null 2>&1; then
  echo ""
  echo "No se encontró 'cloudflared'. Instálalo en segundos:"
  echo "  macOS:  brew install cloudflared"
  echo "  Linux:  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  exit 1
fi

if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
  echo "==> Creando entorno virtual de Python..."
  python3 -m venv "$ROOT_DIR/backend/.venv"
  "$ROOT_DIR/backend/.venv/bin/pip" install --upgrade pip wheel setuptools
  "$ROOT_DIR/backend/.venv/bin/pip" install -r "$ROOT_DIR/backend/requirements.txt"
fi

if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
  echo "==> Instalando dependencias del frontend..."
  (cd "$ROOT_DIR/frontend" && npm install)
fi

echo "==> Compilando interfaz web (optimizada para móvil)..."
(cd "$ROOT_DIR/frontend" && npm run build)

if [ -n "${FRIENDFINDER_PIN:-}" ]; then
  echo "==> Protección por PIN activada (FRIENDFINDER_PIN)."
else
  echo "==> Consejo: puedes proteger el acceso con PIN ejecutando:"
  echo "    FRIENDFINDER_PIN=1234 ./run-cloudflare.sh"
fi

echo "==> Iniciando servidor unificado en http://127.0.0.1:${PORT}..."
(cd "$ROOT_DIR/backend" && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT") &
BACKEND_PID=$!

cleanup() {
  echo ""
  echo "Deteniendo FriendFinder y Cloudflare Tunnel..."
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2

if [ -n "${CLOUDFLARED_TOKEN:-}" ]; then
  echo "==> Conectando túnel permanente de Cloudflare (CLOUDFLARED_TOKEN)..."
  cloudflared tunnel run --token "$CLOUDFLARED_TOKEN"
else
  echo "==> Abriendo túnel HTTPS temporal de Cloudflare (ábrelo en tu móvil)..."
  cloudflared tunnel --url "http://127.0.0.1:${PORT}"
fi
