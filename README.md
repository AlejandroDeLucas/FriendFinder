# FriendFinder

Aplicación web **100% privada** (optimizada para PC y móvil) para analizar fotos y vídeos frame a frame, detectar y agrupar a tus amigos automáticamente, extraer sus mejores capturas en **3 formatos** listos para videojuegos/avatares, descartar automáticamente fotos redundantes o de peor calidad, y buscar en qué fotos o segundos exactos de vídeo aparece cualquier persona a partir de una foto de referencia.

## Qué hace FriendFinder

1. **Análisis frame a frame de fotos y vídeos (100% local)**
   - Reconocimiento facial e embeddings de identidad con **InsightFace (`buffalo_l` / ArcFace)** vía ONNXRuntime.
   - Detección y segmentación por instancia de cuerpo entero con **YOLOv8-seg** (PyTorch).
   - Detecta automáticamente aceleración local: **NVIDIA CUDA**, **Apple Silicon (MPS / CoreML)** o **CPU**.

2. **Los 3 formatos extraídos por persona (Exportables en Pack ZIP para videojuego)**
   - **Retrato en alta calidad**: encuadre de cara + hombros puntuado por nitidez, resolución, frontalidad e iluminación.
   - **Recorte PNG sin fondo ni otras personas**: usa la máscara de segmentación de **solo esa persona** y resta las máscaras del resto de personas del frame, generando un `.png` con canal alfa transparente.
   - **Frame original + recorte limpio**: conserva el frame completo y un recorte rectangular limpio de la persona.

3. **Caducidad automática de 1 hora en lo que subes (conservando todo lo generado)**
   - Por defecto, los archivos originales que subes (fotos/vídeos pesados) se **borran automáticamente tras 1 hora** para no llenar tu disco.
   - **Todo lo generado** (perfiles de amigos, mejores retratos, recortes PNG sin fondo, crops y frames guardados) **se mantiene siempre**.
   - Si quieres conservar un vídeo/foto original (por ejemplo, para reproducirlo y saltar a timestamps), puedes activar **«Mantener originales»** al subirlo o pulsar **«Mantener»** en la Biblioteca antes de que caduque.

4. **Filtrado y descarte inteligente de menor calidad**
   - Al subir un lote de fotos o vídeos, evalúa cada aparición frente a las mejores capturas ya guardadas para cada persona (tanto en retrato como en cuerpo entero).
   - Marca automáticamente las tomas redundantes o borrosas como **Descartadas** con el motivo exacto, pero permite inspeccionarlas y **Rescatar** cualquiera con un clic.

5. **Búsqueda inversa por foto (con saltos exactos en vídeo)**
   - Sube cualquier foto de una persona y FriendFinder te dice en qué perfiles, fotos y **segundos exactos** de cada vídeo aparece, con reproductor integrado para saltar al instante.

6. **Comprensión opcional de escena / actividad (100% local)**
   - Análisis instantáneo sin descargas extra (objetos COCO de YOLO + plano + solo/grupo) y soporte opcional configurable para **BLIP local offline** o **Ollama local**.

---

## Probar desde el móvil con Cloudflare (HTTPS)

> **Nota técnica:** Cloudflare Workers/Pages no pueden ejecutar modelos pesados de visión (PyTorch + InsightFace + decodificación de vídeo) directamente dentro de un Worker JS. Por eso FriendFinder incluye **dos modos Cloudflare** listos para usar:

### Opción A — Cloudflare Tunnel en 1 comando (Recomendado, 100% gratis)
Tu PC/Mac ejecuta la IA (aprovechando tu CPU/GPU gratis) y **Cloudflare Tunnel** te da una URL `https://....trycloudflare.com` instantánea (o tu propio dominio con `CLOUDFLARED_TOKEN`) para abrir la app desde el móvil con interfaz adaptada y barra inferior táctil.

**macOS / Linux:**
```bash
chmod +x run-cloudflare.sh
# Opcional: protege el acceso remoto con un PIN numérico
FRIENDFINDER_PIN=1234 ./run-cloudflare.sh
```

**Windows:**
```bat
set FRIENDFINDER_PIN=1234
run-cloudflare.bat
```

Abre en tu móvil la URL `https://...trycloudflare.com` que imprime `cloudflared` (¡puedes añadirla a la pantalla de inicio como app!).

### Opción B — Cloudflare Pages (Frontend) + Docker / Cloudflare Containers (Backend)
- **Frontend en Cloudflare Pages:** incluye `frontend/wrangler.toml`. Despliega con:
  ```bash
  cd frontend
  npm run build
  npx wrangler pages deploy dist
  ```
  Luego abre **Ajustes → Conexión Cloudflare** en la app e introduce la URL HTTPS de tu backend (túnel o contenedor), o define `VITE_API_BASE_URL` en el build.
- **Contenedor Docker unificado:** incluye `Dockerfile` multi-stage listo para **Cloudflare Containers** o cualquier servidor Docker:
  ```bash
  docker build -t friendfinder .
  docker run -p 8000:8000 -e FRIENDFINDER_PIN=1234 -v $(pwd)/data:/app/data friendfinder
  ```

---

## Uso 100% local (sin Cloudflare)

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

### Windows
Haz doble clic en `run.bat` o ejecútalo desde terminal:
```bat
run.bat
```

Abre tu navegador en **http://127.0.0.1:5173**.
