/**
 * 100% client-side AI engine running directly in the user's mobile/desktop browser:
 * - @vladmandic/face-api: face detection, 68 landmarks, 128-D face identity embeddings, expressions
 * - @mediapipe/tasks-vision: person segmentation (transparent PNG cutouts) + COCO object detection
 */

export interface BrowserFace {
  bbox: [number, number, number, number];
  detScore: number;
  embedding: number[]; // 128-D descriptor
  frontality: number; // 0..100
  expression: string;
}

export interface BrowserBody {
  bbox: [number, number, number, number];
  conf: number;
}

export interface BrowserObject {
  label: string;
  conf: number;
}

export interface BrowserFrameAnalysis {
  faces: BrowserFace[];
  bodies: BrowserBody[];
  objects: BrowserObject[];
  personMask: Float32Array | null; // per-pixel foreground confidence (0..1)
  maskWidth: number;
  maskHeight: number;
}

export interface BrowserQualityMetrics {
  portrait_score: number;
  body_score: number;
  overall_score: number;
  sharpness_score: number;
  frontality_score: number;
  separation_score: number;
  is_full_body: boolean;
  matchedBody: BrowserBody | null;
  matchedBodyIdx: number | null;
}

const EXPRESSION_ES: Record<string, string> = {
  happy: 'sonriendo',
  surprised: 'sorprendido/a',
  sad: 'pensativo/a',
  angry: 'serio/a',
  neutral: 'expresión relajada',
};

const COCO_ES: Record<string, string> = {
  bicycle: 'en bicicleta',
  car: 'junto a un coche',
  motorcycle: 'en moto',
  bus: 'en autobús',
  bench: 'en un banco',
  cat: 'con un gato',
  dog: 'con un perro',
  backpack: 'con mochila',
  umbrella: 'con paraguas',
  handbag: 'con bolso',
  suitcase: 'de viaje',
  'sports ball': 'con balón',
  skateboard: 'en skate',
  surfboard: 'haciendo surf',
  bottle: 'con bebida',
  'wine glass': 'con copa',
  cup: 'tomando algo',
  pizza: 'comiendo pizza',
  cake: 'celebrando con tarta',
  chair: 'junto a sillas',
  couch: 'en el sofá',
  'dining table': 'en una mesa',
  laptop: 'usando portátil',
  'cell phone': 'con el móvil',
  book: 'leyendo',
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let faceApiMod: any = null;
let faceModelsReady = false;
let faceModelsPromise: Promise<void> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let segmenterInstance: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let objectDetectorInstance: any = null;
let mediaPipeTried = false;
let mediaPipePromise: Promise<void> | null = null;

async function ensureFaceModels(onStatus?: (msg: string) => void): Promise<void> {
  if (faceModelsReady && faceApiMod) return;
  if (faceModelsPromise) return faceModelsPromise;

  faceModelsPromise = (async () => {
    onStatus?.('Cargando IA facial en el navegador (solo la primera vez)...');
    const faceapi = await import('@vladmandic/face-api');
    faceApiMod = faceapi;

    // Prefer Cloudflare edge /models, fallback to jsDelivr CDN
    const sources = [
      '/models',
      'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model',
    ];

    let loaded = false;
    let lastErr: unknown = null;
    for (const baseUrl of sources) {
      try {
        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(baseUrl),
          faceapi.nets.tinyFaceDetector.loadFromUri(baseUrl),
          faceapi.nets.faceLandmark68TinyNet.loadFromUri(baseUrl),
          faceapi.nets.faceRecognitionNet.loadFromUri(baseUrl),
          faceapi.nets.faceExpressionNet.loadFromUri(baseUrl),
        ]);
        loaded = true;
        break;
      } catch (err) {
        lastErr = err;
      }
    }
    if (!loaded) {
      throw new Error(`No se pudieron cargar los modelos faciales: ${String(lastErr)}`);
    }
    faceModelsReady = true;
  })();

  try {
    await faceModelsPromise;
  } catch (err) {
    faceModelsPromise = null;
    throw err;
  }
}

async function ensureMediaPipe(onStatus?: (msg: string) => void): Promise<void> {
  if (mediaPipeTried) return;
  if (mediaPipePromise) return mediaPipePromise;

  mediaPipePromise = (async () => {
    try {
      onStatus?.('Cargando segmentación PNG sin fondo...');
      const vision = await import('@mediapipe/tasks-vision');
      let fileset = null;
      for (const wasmBase of [
        '/wasm',
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm',
      ]) {
        try {
          fileset = await vision.FilesetResolver.forVisionTasks(wasmBase);
          break;
        } catch {
          // try fallback
        }
      }
      if (!fileset) {
        mediaPipeTried = true;
        return;
      }

      // Tiny 250KB selfie/person segmenter for transparent PNG cutouts
      try {
        segmenterInstance = await vision.ImageSegmenter.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
          },
          runningMode: 'IMAGE',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        });
      } catch {
        segmenterInstance = null;
      }

      // Lightweight COCO object detector for individual person boxes + context objects
      try {
        objectDetectorInstance = await vision.ObjectDetector.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
          },
          runningMode: 'IMAGE',
          scoreThreshold: 0.3,
          maxResults: 20,
        });
      } catch {
        objectDetectorInstance = null;
      }
    } catch {
      // Graceful fallback: face detection/clustering/portraits still work 100%
    } finally {
      mediaPipeTried = true;
    }
  })();

  await mediaPipePromise;
}

function boxIoU(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ox = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const oy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ox * oy;
  if (inter <= 0) return 0;
  const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
  const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
  return inter / (areaA + areaB - inter);
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapRawFaceItem(item: any, w: number, h: number): BrowserFace {
  const box = item.detection.box;
  const x1 = Math.max(0, Math.min(w - 1, box.x));
  const y1 = Math.max(0, Math.min(h - 1, box.y));
  const x2 = Math.max(x1 + 1, Math.min(w, box.x + box.width));
  const y2 = Math.max(y1 + 1, Math.min(h, box.y + box.height));

  // Estimate frontality from 68-point eyes + nose symmetry
  let frontality = 75;
  try {
    const leftEye = item.landmarks.getLeftEye();
    const rightEye = item.landmarks.getRightEye();
    const nose = item.landmarks.getNose();
    const lx = leftEye.reduce((s: number, p: { x: number }) => s + p.x, 0) / leftEye.length;
    const ly = leftEye.reduce((s: number, p: { y: number }) => s + p.y, 0) / leftEye.length;
    const rx = rightEye.reduce((s: number, p: { x: number }) => s + p.x, 0) / rightEye.length;
    const ry = rightEye.reduce((s: number, p: { y: number }) => s + p.y, 0) / rightEye.length;
    const nx = nose[Math.floor(nose.length / 2)].x;
    const eyeDist = Math.hypot(rx - lx, ry - ly);
    if (eyeDist > 2) {
      const midX = 0.5 * (lx + rx);
      const horizOffset = Math.abs(nx - midX) / eyeDist;
      const tilt = Math.abs(ry - ly) / eyeDist;
      const penalty = Math.min(1, horizOffset * 1.45 + tilt * 0.8);
      frontality = Math.max(15, Math.min(100, 100 * (1 - penalty)));
    }
  } catch {
    frontality = 75;
  }

  let expression = 'neutral';
  if (item.expressions) {
    let bestVal = 0;
    for (const [k, v] of Object.entries(item.expressions as Record<string, number>)) {
      if (v > bestVal) {
        bestVal = v;
        expression = k;
      }
    }
  }

  return {
    bbox: [x1, y1, x2, y2],
    detScore: Number(item.detection.score || 0.7),
    embedding: Array.from(item.descriptor as Float32Array),
    frontality: Math.round(frontality * 10) / 10,
    expression,
  };
}

export async function detectFacesOnlyBrowser(
  canvas: HTMLCanvasElement,
  onStatus?: (msg: string) => void
): Promise<BrowserFace[]> {
  await ensureFaceModels(onStatus);
  const faceapi = faceApiMod;
  const w = canvas.width;
  const h = canvas.height;

  // Primary detector: SSD MobileNet v1 with sensitive threshold for dim/party/group scenes
  const ssdOpts = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.28, maxResults: 30 });
  const rawSsd = await faceapi
    .detectAllFaces(canvas, ssdOpts)
    .withFaceLandmarks(true)
    .withFaceExpressions()
    .withFaceDescriptors();

  // Secondary detector: TinyFaceDetector catches smaller or angled people that SSD misses
  const tinyOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.34 });
  const rawTiny = await faceapi
    .detectAllFaces(canvas, tinyOpts)
    .withFaceLandmarks(true)
    .withFaceExpressions()
    .withFaceDescriptors();

  const merged: BrowserFace[] = [];
  for (const item of rawSsd) {
    merged.push(mapRawFaceItem(item, w, h));
  }
  for (const item of rawTiny) {
    const candidate = mapRawFaceItem(item, w, h);
    // Ignore tiny background false positives (< 26px)
    if (Math.min(candidate.bbox[2] - candidate.bbox[0], candidate.bbox[3] - candidate.bbox[1]) < 26) {
      continue;
    }
    const overlapsExisting = merged.some((m) => boxIoU(m.bbox, candidate.bbox) > 0.38);
    if (!overlapsExisting) {
      merged.push(candidate);
    }
  }

  return merged;
}

export async function analyzeFrameBrowser(
  canvas: HTMLCanvasElement,
  onStatus?: (msg: string) => void
): Promise<BrowserFrameAnalysis> {
  const faces = await detectFacesOnlyBrowser(canvas, onStatus);
  if (faces.length === 0) {
    return { faces: [], bodies: [], objects: [], personMask: null, maskWidth: 0, maskHeight: 0 };
  }

  await ensureMediaPipe(onStatus);

  const bodies: BrowserBody[] = [];
  const objects: BrowserObject[] = [];
  let personMask: Float32Array | null = null;
  let maskWidth = 0;
  let maskHeight = 0;

  if (objectDetectorInstance) {
    try {
      const detRes = objectDetectorInstance.detect(canvas);
      for (const d of detRes.detections || []) {
        const cat = d.categories?.[0];
        const bb = d.boundingBox;
        if (!cat || !bb) continue;
        const label = String(cat.categoryName || '').toLowerCase();
        const conf = Number(cat.score || 0);
        const bbox: [number, number, number, number] = [
          Math.max(0, bb.originX),
          Math.max(0, bb.originY),
          Math.min(canvas.width, bb.originX + bb.width),
          Math.min(canvas.height, bb.originY + bb.height),
        ];
        if (label === 'person') {
          bodies.push({ bbox, conf });
        } else {
          objects.push({ label, conf });
        }
      }
    } catch {
      // ignore optional detector error
    }
  }

  if (segmenterInstance) {
    try {
      const segRes = segmenterInstance.segment(canvas);
      const masks = segRes.confidenceMasks;
      if (masks && masks.length > 0) {
        // Selfie segmenter: index 0 or 1 depending on model; check class with foreground
        const maskObj = masks.length > 1 ? masks[1] : masks[0];
        maskWidth = maskObj.width;
        maskHeight = maskObj.height;
        personMask = new Float32Array(maskObj.getAsFloat32Array());
      }
      segRes.close?.();
    } catch {
      personMask = null;
    }
  }

  return { faces, bodies, objects, personMask, maskWidth, maskHeight };
}

export function euclideanDistance(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/** Convert 128-D face-api Euclidean distance into an intuitive 0..1 similarity score */
export function descriptorSimilarity(a: number[], b: number[]): number {
  const dist = euclideanDistance(a, b);
  return Math.max(0, Math.min(1, 1 - dist / 1.15));
}

function computeSharpnessAndLighting(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number]
): { sharpness: number; lighting: number } {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const x1 = Math.max(0, Math.floor(bbox[0]));
  const y1 = Math.max(0, Math.floor(bbox[1]));
  const bw = Math.max(4, Math.min(w - x1, Math.ceil(bbox[2] - x1)));
  const bh = Math.max(4, Math.min(h - y1, Math.ceil(bbox[3] - y1)));

  // Downsample to <= 96px for fast, resolution-normalized Laplacian variance
  const scale = Math.min(1, 96 / Math.max(bw, bh));
  const sw = Math.max(8, Math.round(bw * scale));
  const sh = Math.max(8, Math.round(bh * scale));

  const tmp = document.createElement('canvas');
  tmp.width = sw;
  tmp.height = sh;
  const tctx = tmp.getContext('2d');
  if (!tctx) return { sharpness: 60, lighting: 70 };
  tctx.drawImage(ctx.canvas, x1, y1, bw, bh, 0, 0, sw, sh);
  const data = tctx.getImageData(0, 0, sw, sh).data;

  const gray = new Float32Array(sw * sh);
  let mean = 0;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const lum = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
    gray[i] = lum;
    mean += lum;
  }
  mean /= gray.length;

  let varSum = 0;
  for (let i = 0; i < gray.length; i++) {
    const d = gray[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / gray.length);

  // Discrete 4-neighbor Laplacian variance
  let lapSum = 0;
  let lapSqSum = 0;
  let count = 0;
  for (let y = 1; y < sh - 1; y++) {
    for (let x = 1; x < sw - 1; x++) {
      const idx = y * sw + x;
      const lap =
        gray[idx - 1] + gray[idx + 1] + gray[idx - sw] + gray[idx + sw] - 4 * gray[idx];
      lapSum += lap;
      lapSqSum += lap * lap;
      count++;
    }
  }
  const lapMean = count > 0 ? lapSum / count : 0;
  const lapVar = count > 0 ? Math.max(0, lapSqSum / count - lapMean * lapMean) : 0;
  const sharpness = Math.max(0, Math.min(100, 100 * (1 - Math.exp(-lapVar / 130))));

  let brightScore = 100;
  if (mean < 85) brightScore = Math.max(0, (mean / 85) * 100);
  else if (mean > 185) brightScore = Math.max(0, (1 - (mean - 185) / 70) * 100);
  const contrastScore = Math.min(100, (std / 38) * 100);
  const lighting = 0.65 * brightScore + 0.35 * contrastScore;

  return { sharpness, lighting };
}

function matchFaceToBodyBrowser(
  face: BrowserFace,
  bodies: BrowserBody[],
  used: Set<number>
): { idx: number | null; body: BrowserBody | null } {
  const [fx1, fy1, fx2, fy2] = face.bbox;
  const fcx = 0.5 * (fx1 + fx2);
  const fcy = 0.5 * (fy1 + fy2);
  const fw = Math.max(1, fx2 - fx1);
  const fh = Math.max(1, fy2 - fy1);

  let bestIdx: number | null = null;
  let bestScore = -1;

  bodies.forEach((b, idx) => {
    if (used.has(idx)) return;
    const [bx1, by1, bx2, by2] = b.bbox;
    const bw = Math.max(1, bx2 - bx1);
    const bh = Math.max(1, by2 - by1);

    if (fcx < bx1 - 0.15 * fw || fcx > bx2 + 0.15 * fw) return;
    if (fcy < by1 - 0.25 * fh || fcy > by1 + 0.58 * bh) return;

    const ox = Math.max(0, Math.min(fx2, bx2) - Math.max(fx1, bx1));
    const oy = Math.max(0, Math.min(fy2, by2) - Math.max(fy1, by1));
    const overlap = (ox * oy) / (fw * fh);
    if (overlap < 0.4) return;

    const hCenter = 1 - Math.min(1, Math.abs(fcx - 0.5 * (bx1 + bx2)) / (0.6 * bw));
    const vPos = 1 - Math.min(1, Math.abs((fcy - by1) / bh - 0.16) / 0.45);
    const score = 0.5 * overlap + 0.3 * hCenter + 0.2 * vPos;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });

  return { idx: bestIdx, body: bestIdx !== null ? bodies[bestIdx] : null };
}

export function scoreDetectionBrowser(
  canvas: HTMLCanvasElement,
  face: BrowserFace,
  bodies: BrowserBody[],
  hasMask: boolean,
  usedBodies: Set<number>
): BrowserQualityMetrics {
  const ctx = canvas.getContext('2d')!;
  const [fx1, fy1, fx2, fy2] = face.bbox;
  const fw = Math.max(1, fx2 - fx1);
  const fh = Math.max(1, fy2 - fy1);
  const minDim = Math.min(fw, fh);

  const { sharpness, lighting } = computeSharpnessAndLighting(ctx, face.bbox);
  const resScore = Math.min(100, Math.pow(minDim / 150, 0.75) * 100);
  const detScore = Math.min(100, Math.max(0, ((face.detScore - 0.25) / 0.7) * 100));

  const margin = Math.max(2, Math.floor(0.01 * Math.min(canvas.width, canvas.height)));
  const cutOff =
    fx1 <= margin ||
    fy1 <= margin ||
    fx2 >= canvas.width - margin ||
    fy2 >= canvas.height - margin;

  const portraitScore =
    (cutOff ? 0.8 : 1) *
    (0.32 * sharpness +
      0.25 * resScore +
      0.23 * face.frontality +
      0.1 * detScore +
      0.1 * lighting);

  const { idx, body } = matchFaceToBodyBrowser(face, bodies, usedBodies);
  if (!body || idx === null) {
    return {
      portrait_score: Math.round(portraitScore * 10) / 10,
      body_score: hasMask ? Math.round(portraitScore * 0.55 * 10) / 10 : 0,
      overall_score: Math.round(portraitScore * 10) / 10,
      sharpness_score: Math.round(sharpness * 10) / 10,
      frontality_score: face.frontality,
      separation_score: bodies.length <= 1 ? 100 : 65,
      is_full_body: false,
      matchedBody: null,
      matchedBodyIdx: null,
    };
  }

  const [bx1, by1, bx2, by2] = body.bbox;
  const bw = Math.max(1, bx2 - bx1);
  const bh = Math.max(1, by2 - by1);
  const targetArea = bw * bh;

  let maxOverlap = 0;
  bodies.forEach((other, j) => {
    if (j === idx) return;
    const ox = Math.max(0, Math.min(bx2, other.bbox[2]) - Math.max(bx1, other.bbox[0]));
    const oy = Math.max(0, Math.min(by2, other.bbox[3]) - Math.max(by1, other.bbox[1]));
    maxOverlap = Math.max(maxOverlap, (ox * oy) / targetArea);
  });
  const separation = Math.max(0, Math.min(100, 100 * (1 - Math.min(1, maxOverlap * 1.35))));

  const bodyEdgeClipped = bx1 <= margin || bx2 >= canvas.width - margin || cutOff;
  const bodyEdgeMult = bodyEdgeClipped ? 0.78 : 1.0;

  const bodyToFace = bh / fh;
  const aspect = bh / bw;
  const isFullBody = bodyToFace >= 3.4 && aspect >= 1.3;
  const bodyRes = Math.min(100, Math.pow(bh / 380, 0.7) * 100);
  const coverage = Math.min(100, (bodyToFace / 5.2) * 100);
  const bodyScore =
    bodyEdgeMult *
    (0.24 * bodyRes +
      0.24 * coverage +
      0.24 * separation +
      0.18 * portraitScore +
      0.1 * (hasMask ? 100 : 40));

  const overall = Math.max(portraitScore, 0.58 * portraitScore + 0.42 * bodyScore);

  return {
    portrait_score: Math.round(portraitScore * 10) / 10,
    body_score: Math.round(bodyScore * 10) / 10,
    overall_score: Math.round(overall * 10) / 10,
    sharpness_score: Math.round(sharpness * 10) / 10,
    frontality_score: face.frontality,
    separation_score: Math.round(separation * 10) / 10,
    is_full_body: isFullBody,
    matchedBody: body,
    matchedBodyIdx: idx,
  };
}

export function buildBrowserSceneSummary(
  face: BrowserFace,
  numPeople: number,
  isFullBody: boolean,
  objects: BrowserObject[]
): { summary: string; tags: string[] } {
  const parts: string[] = [];
  const tags: string[] = [];

  parts.push(isFullBody ? 'Plano de cuerpo entero' : 'Retrato / plano cercano');
  tags.push(isFullBody ? 'cuerpo entero' : 'retrato');

  const exprEs = EXPRESSION_ES[face.expression];
  if (exprEs) {
    parts.push(exprEs);
    tags.push(exprEs);
  }

  if (numPeople <= 1) {
    parts.push('solo/a en la escena');
    tags.push('solo');
  } else if (numPeople === 2) {
    parts.push('acompañado/a (2 personas)');
    tags.push('dúo');
  } else {
    parts.push(`en grupo (${numPeople} personas)`);
    tags.push('grupo');
  }

  const seen = new Set<string>();
  const ctx: string[] = [];
  for (const obj of [...objects].sort((a, b) => b.conf - a.conf)) {
    if (seen.has(obj.label) || obj.conf < 0.42) continue;
    seen.add(obj.label);
    if (COCO_ES[obj.label]) {
      ctx.push(COCO_ES[obj.label]);
      tags.push(obj.label);
    }
    if (ctx.length >= 2) break;
  }
  if (ctx.length > 0) parts.push(ctx.join(', '));

  return { summary: parts.join(' · '), tags: tags.slice(0, 8) };
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality = 0.92): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('No se pudo codificar la imagen'))),
      type,
      quality
    );
  });
}

function cropCanvas(
  source: HTMLCanvasElement,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  maxDim?: number
): HTMLCanvasElement {
  const sx = Math.max(0, Math.floor(x1));
  const sy = Math.max(0, Math.floor(y1));
  const sw = Math.max(1, Math.min(source.width - sx, Math.ceil(x2 - sx)));
  const sh = Math.max(1, Math.min(source.height - sy, Math.ceil(y2 - sy)));
  const scale = maxDim ? Math.min(1, maxDim / Math.max(sw, sh)) : 1;
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));

  const out = document.createElement('canvas');
  out.width = dw;
  out.height = dh;
  out.getContext('2d')!.drawImage(source, sx, sy, sw, sh, 0, 0, dw, dh);
  return out;
}

export function computePortraitBoxBrowser(
  faceBbox: [number, number, number, number],
  imgW: number,
  imgH: number
): [number, number, number, number] {
  const [fx1, fy1, fx2, fy2] = faceBbox;
  const fw = Math.max(1, fx2 - fx1);
  const fh = Math.max(1, fy2 - fy1);
  const cx = 0.5 * (fx1 + fx2);
  return [
    Math.max(0, cx - 1.15 * fw),
    Math.max(0, fy1 - 0.75 * fh),
    Math.min(imgW, cx + 1.15 * fw),
    Math.min(imgH, fy2 + 1.25 * fh),
  ];
}

export async function extractThumbBlobBrowser(
  source: HTMLCanvasElement,
  faceBbox: [number, number, number, number]
): Promise<Blob> {
  const [px1, py1, px2, py2] = computePortraitBoxBrowser(faceBbox, source.width, source.height);
  const c = cropCanvas(source, px1, py1, px2, py2, 320);
  return canvasToBlob(c, 'image/jpeg', 0.82);
}

/** Sample a Float32 mask with smooth bilinear interpolation (eliminates jagged/blocky edges). */
function sampleMaskBilinear(
  mask: Float32Array,
  mw: number,
  mh: number,
  u: number,
  v: number
): number {
  const x = Math.max(0, Math.min(mw - 1, u * (mw - 1)));
  const y = Math.max(0, Math.min(mh - 1, v * (mh - 1)));
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(mw - 1, x0 + 1);
  const y1 = Math.min(mh - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;

  const v00 = mask[y0 * mw + x0];
  const v10 = mask[y0 * mw + x1];
  const v01 = mask[y1 * mw + x0];
  const v11 = mask[y1 * mw + x1];

  return (
    v00 * (1 - fx) * (1 - fy) +
    v10 * fx * (1 - fy) +
    v01 * (1 - fx) * fy +
    v11 * fx * fy
  );
}

/** Distance from point (x, y) to a vertical spine segment representing a person's head-to-torso axis. */
function distToPersonSpine(
  x: number,
  y: number,
  bbox: [number, number, number, number]
): number {
  const cx = 0.5 * (bbox[0] + bbox[2]);
  const bw = Math.max(20, bbox[2] - bbox[0]);
  const yTop = bbox[1] + 0.1 * (bbox[3] - bbox[1]);
  const yBot = bbox[3] - 0.1 * (bbox[3] - bbox[1]);
  const clampedY = Math.max(yTop, Math.min(yBot, y));
  // Normalize horizontal distance by person width so narrow/wide people compete fairly
  const dx = (x - cx) / (0.65 * bw);
  const dy = (y - clampedY) / Math.max(40, bbox[3] - bbox[1]);
  return Math.hypot(dx, dy);
}

/** Keep only foreground pixels connected to this person's face/chest seed so strangers' limbs vanish organically. */
function keepConnectedToSeed(
  alpha: Uint8Array,
  w: number,
  h: number,
  seedX: number,
  seedY: number
): Uint8Array {
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;

  const sx = Math.max(0, Math.min(w - 1, Math.round(seedX)));
  const sy = Math.max(0, Math.min(h - 1, Math.round(seedY)));

  // Seed a small disk around face/upper-chest center so BFS always catches the person's mask
  const r = Math.max(6, Math.round(Math.min(w, h) * 0.04));
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const nx = sx + dx;
      const ny = sy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const idx = ny * w + nx;
      if (!visited[idx] && alpha[idx] > 35) {
        visited[idx] = 1;
        queue[tail++] = idx;
      }
    }
  }

  if (tail === 0) return alpha; // fallback if seed missed

  while (head < tail) {
    const idx = queue[head++];
    const x = idx % w;
    const y = (idx - x) / w;

    if (x > 0) {
      const n = idx - 1;
      if (!visited[n] && alpha[n] > 28) {
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
    if (x + 1 < w) {
      const n = idx + 1;
      if (!visited[n] && alpha[n] > 28) {
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
    if (y > 0) {
      const n = idx - w;
      if (!visited[n] && alpha[n] > 28) {
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
    if (y + 1 < h) {
      const n = idx + w;
      if (!visited[n] && alpha[n] > 28) {
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
  }

  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) {
    out[i] = visited[i] ? alpha[i] : 0;
  }
  return out;
}

/** Smooth 3x3 Gaussian-like alpha feathering for natural, non-jagged organic silhouette edges. */
function featherAlpha(alpha: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const yUp = Math.max(0, y - 1) * w;
    const yMid = y * w;
    const yDn = Math.min(h - 1, y + 1) * w;
    for (let x = 0; x < w; x++) {
      const xL = Math.max(0, x - 1);
      const xR = Math.min(w - 1, x + 1);
      const val =
        alpha[yUp + xL] +
        2 * alpha[yUp + x] +
        alpha[yUp + xR] +
        2 * alpha[yMid + xL] +
        4 * alpha[yMid + x] +
        2 * alpha[yMid + xR] +
        alpha[yDn + xL] +
        2 * alpha[yDn + x] +
        alpha[yDn + xR];
      out[yMid + x] = Math.round(val / 16);
    }
  }
  return out;
}

export async function extractAllAssetBlobsBrowser(
  source: HTMLCanvasElement,
  faceBbox: [number, number, number, number],
  matchedBody: BrowserBody | null,
  allBodies: BrowserBody[],
  personMask: Float32Array | null,
  maskW: number,
  maskH: number
): Promise<{
  frameBlob: Blob;
  portraitBlob: Blob;
  cropBlob: Blob;
  thumbBlob: Blob;
  cutoutBlob: Blob | null;
}> {
  const w = source.width;
  const h = source.height;
  const [px1, py1, px2, py2] = computePortraitBoxBrowser(faceBbox, w, h);

  const portraitCanvas = cropCanvas(source, px1, py1, px2, py2);
  const thumbCanvas = cropCanvas(source, px1, py1, px2, py2, 360);

  // Generous working region around the person so arms, hair, and shoulders are NEVER clipped square
  let cx1 = px1;
  let cy1 = py1;
  let cx2 = px2;
  let cy2 = py2;
  if (matchedBody) {
    const [bx1, by1, bx2, by2] = matchedBody.bbox;
    const padX = 0.14 * (bx2 - bx1);
    const padYTop = 0.1 * (by2 - by1);
    const padYBot = 0.08 * (by2 - by1);
    cx1 = Math.max(0, bx1 - padX);
    cy1 = Math.max(0, by1 - padYTop);
    cx2 = Math.min(w, bx2 + padX);
    cy2 = Math.min(h, by2 + padYBot);
  } else {
    // Expand downward to capture upper body/shoulders even if object detector didn't catch body box
    const fw = Math.max(1, faceBbox[2] - faceBbox[0]);
    const fh = Math.max(1, faceBbox[3] - faceBbox[1]);
    const fcx = 0.5 * (faceBbox[0] + faceBbox[2]);
    cx1 = Math.max(0, fcx - 1.7 * fw);
    cy1 = Math.max(0, faceBbox[1] - 0.9 * fh);
    cx2 = Math.min(w, fcx + 1.7 * fw);
    cy2 = Math.min(h, faceBbox[3] + 3.0 * fh);
  }
  const cropCanvasEl = cropCanvas(source, cx1, cy1, cx2, cy2);

  // Organic Transparent PNG Cutout (bilinear mask + soft Voronoi separation + connected component + feathering)
  let cutoutBlob: Blob | null = null;
  if (personMask && maskW > 0 && maskH > 0) {
    const workCanvas = cropCanvas(source, cx1, cy1, cx2, cy2);
    const cw = workCanvas.width;
    const ch = workCanvas.height;
    const workCtx = workCanvas.getContext('2d')!;
    const imgData = workCtx.getImageData(0, 0, cw, ch);
    const data = imgData.data;

    // Auto-check mask polarity at the person's face center so mask is never inverted
    const fcx = 0.5 * (faceBbox[0] + faceBbox[2]);
    const fcy = 0.5 * (faceBbox[1] + faceBbox[3]);
    const faceMaskVal = sampleMaskBilinear(personMask, maskW, maskH, fcx / w, fcy / h);
    const invertMask = faceMaskVal < 0.35;

    const targetBox: [number, number, number, number] = matchedBody
      ? matchedBody.bbox
      : [cx1, cy1, cx2, cy2];
    const otherBoxes = allBodies
      .filter((b) => b !== matchedBody)
      .map((b) => b.bbox);

    const sx0 = Math.floor(cx1);
    const sy0 = Math.floor(cy1);
    const rawAlpha = new Uint8Array(cw * ch);

    for (let y = 0; y < ch; y++) {
      const gy = sy0 + y;
      const v = gy / Math.max(1, h);
      for (let x = 0; x < cw; x++) {
        const gx = sx0 + x;
        const u = gx / Math.max(1, w);
        let conf = sampleMaskBilinear(personMask, maskW, maskH, u, v);
        if (invertMask) conf = 1 - conf;

        // Smooth organic separation from other people via spine distance (no hard rectangular cuts!)
        if (otherBoxes.length > 0) {
          const dMine = distToPersonSpine(gx, gy, targetBox);
          let dOtherMin = Infinity;
          for (const ob of otherBoxes) {
            const d = distToPersonSpine(gx, gy, ob);
            if (d < dOtherMin) dOtherMin = d;
          }
          // Soft sigmoid ownership weight: 1.0 near our spine, fades smoothly to 0 near other person's spine
          const marginDiff = dOtherMin - dMine;
          if (marginDiff < 0.25) {
            const ownership = Math.max(0, Math.min(1, (marginDiff + 0.18) / 0.43));
            conf *= ownership;
          }
        }

        // Smooth alpha curve (gentle organic contour)
        const a =
          conf >= 0.62
            ? 255
            : conf <= 0.24
              ? 0
              : Math.round(((conf - 0.24) / 0.38) * 255);
        rawAlpha[y * cw + x] = a;
      }
    }

    // Keep only silhouette connected to this person's face/torso + soft edge feathering
    const connected = keepConnectedToSeed(rawAlpha, cw, ch, fcx - sx0, fcy - sy0);
    const smoothAlpha = featherAlpha(connected, cw, ch);

    // Find tight organic bounding box of non-transparent pixels
    let minX = cw;
    let minY = ch;
    let maxX = 0;
    let maxY = 0;
    let fgPixels = 0;

    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        const a = smoothAlpha[y * cw + x];
        data[(y * cw + x) * 4 + 3] = a;
        if (a > 24) {
          fgPixels++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (fgPixels > 160 && maxX > minX + 12 && maxY > minY + 12) {
      workCtx.putImageData(imgData, 0, 0);
      const pad = 16;
      const tx = Math.max(0, minX - pad);
      const ty = Math.max(0, minY - pad);
      const tw = Math.min(cw - tx, maxX - minX + 1 + pad * 2);
      const th = Math.min(ch - ty, maxY - minY + 1 + pad * 2);

      const finalCut = document.createElement('canvas');
      finalCut.width = tw;
      finalCut.height = th;
      finalCut.getContext('2d')!.drawImage(workCanvas, tx, ty, tw, th, 0, 0, tw, th);
      cutoutBlob = await canvasToBlob(finalCut, 'image/png');
    }
  }

  const [frameBlob, portraitBlob, cropBlob, thumbBlob] = await Promise.all([
    canvasToBlob(source, 'image/jpeg', 0.9),
    canvasToBlob(portraitCanvas, 'image/jpeg', 0.94),
    canvasToBlob(cropCanvasEl, 'image/jpeg', 0.92),
    canvasToBlob(thumbCanvas, 'image/jpeg', 0.84),
  ]);

  return { frameBlob, portraitBlob, cropBlob, thumbBlob, cutoutBlob };
}
