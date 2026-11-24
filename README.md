# FriendFinder

Aplicación web **100% local y privada** para analizar fotos y vídeos frame a frame, detectar y agrupar a tus amigos automáticamente, extraer sus mejores capturas en **3 formatos** listos para videojuegos/avatares, descartar automáticamente fotos redundantes o de peor calidad, y buscar en qué fotos o segundos exactos de vídeo aparece cualquier persona a partir de una foto de referencia.

## Qué hace FriendFinder

1. **Análisis frame a frame de fotos y vídeos (100% local)**
   - Reconocimiento facial e embeddings de identidad con **InsightFace (`buffalo_l` / ArcFace)** vía ONNXRuntime.
   - Detección y segmentación por instancia de cuerpo entero con **YOLOv8-seg** (PyTorch).
   - Detecta automáticamente aceleración local: **NVIDIA CUDA**, **Apple Silicon (MPS / CoreML)** o **CPU**.

2. **Los 3 formatos extraídos por persona (Exportables en Pack ZIP para videojuego)**
   - **Retrato en alta calidad**: encuadre de cara + hombros puntuado por nitidez, resolución, frontalidad e iluminación.
   - **Recorte PNG sin fondo ni otras personas**: usa la máscara de segmentación de **solo esa persona** y resta las máscaras del resto de personas del frame, generando un `.png` con canal alfa transparente.
   - **Frame original + recorte limpio**: conserva el frame completo y un recorte rectangular limpio de la persona.

3. **Filtrado y descarte inteligente de menor calidad**
   - Al subir un lote de fotos o vídeos, evalúa cada aparición frente a las mejores capturas ya guardadas para cada persona (tanto en retrato como en cuerpo entero).
   - Marca automáticamente las tomas redundantes o borrosas como **Descartadas** con el motivo exacto, pero permite inspeccionarlas y **Rescatar** cualquiera con un clic.

4. **Búsqueda inversa por foto (con saltos exactos en vídeo)**
   - Mientras mantengas los vídeos en la biblioteca, puedes subir cualquier foto de una persona y FriendFinder te dice en qué fotos y en qué **segundos exactos** de cada vídeo aparece, con reproductor integrado para saltar al instante.
   - Si borras un vídeo con *"Liberar espacio"*, el vídeo se elimina del disco pero **todos los retratos y recortes PNG extraídos se conservan**.

5. **Comprensión opcional de escena / actividad (100% local)**
   - Incluye análisis instantáneo de contexto (objetos detectados + tipo de plano + solo/grupo) sin descargas extra.
   - Opcionalmente activable bajo demanda o en Ajustes: modelo local **BLIP** offline (`Salesforce/blip-image-captioning-base`) o **Ollama local** (`llava`, `moondream`, `llama3.2-vision`).

---

## Requisitos

- **Python 3.10 – 3.12**
- **Node.js 20+**
- *(Opcional)* GPU NVIDIA con CUDA o Mac con Apple Silicon para acelerar vídeos largos.

---

## Arranque rápido

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

> **Nota sobre la primera ejecución:** La primera vez que analices una foto o vídeo, InsightFace (`buffalo_l`) y `yolov8n-seg.pt` se descargarán automáticamente en `data/models/` (todo lo demás ocurre 100% offline en tu equipo).

---

## Instalación manual (desarrollo)

### Backend (FastAPI + ML)
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Opcional: si tienes GPU NVIDIA CUDA, instala torch con CUDA y onnxruntime-gpu:
# pip install onnxruntime-gpu

uvicorn app.main:app --reload --port 8000
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

---

## Estructura del proyecto

```
FriendFinder/
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                  # Entrada FastAPI + archivos estáticos
│   │   ├── config.py                # Rutas locales y detección CUDA/MPS/CPU
│   │   ├── database.py              # Repositorio SQLite (personas, media, detecciones, jobs)
│   │   ├── routers/                 # Endpoints REST + streaming Range de vídeo
│   │   └── services/
│   │       ├── ml_models.py         # InsightFace + YOLOv8-seg
│   │       ├── quality.py           # Puntuación de retrato, cuerpo entero y separación
│   │       ├── extractor.py         # Generación de retrato, PNG sin otros, crop y frame
│   │       ├── pipeline.py          # Ingesta frame a frame, clustering y descarte inteligente
│   │       ├── search.py            # Búsqueda inversa por foto con timestamps
│   │       ├── scene_analyzer.py    # Análisis local de escena (Heurística / BLIP / Ollama)
│   │       └── exporter.py          # Generador de Pack ZIP para videojuegos
│   └── tests/
│       └── test_pipeline.py         # Tests automatizados end-to-end
├── frontend/                        # App React + TypeScript + Vanilla CSS
├── run.sh / run.bat                 # Lanzadores locales en un comando
└── data/                            # Generado automáticamente (DB, modelos y capturas)
```
