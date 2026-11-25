/**
 * Fast, full-resolution client-side Video Frame Extractor.
 * Supports: First & Last frame (EOS-safe), Current frame, Interactive Filmstrip,
 * Anti-Blur Sharpest Frame Finder, Format/Aspect Conversion (PNG/JPG/WebP + 16:9/9:16/1:1), and Batch ZIP.
 */
import JSZip from 'jszip';

export type FrameFormat = 'png' | 'jpeg' | 'webp';
export type FrameAspect = 'original' | '16:9' | '9:16' | '1:1';

export interface ExtractedFrame {
  id: string;
  kind: 'first' | 'last' | 'sharpest' | 'cutout' | 'custom';
  label: string;
  timestampSec: number;
  width: number;
  height: number;
  blob: Blob;
  url: string;
  filename: string;
  isCutout?: boolean;
}

export interface FilmstripThumb {
  timestampSec: number;
  url: string;
}

function stripExt(filename: string): string {
  return filename.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ _-]+/g, '').trim() || 'video';
}

function formatStampForFile(sec: number): string {
  const s = Math.max(0, sec);
  const mins = Math.floor(s / 60);
  const rem = (s % 60).toFixed(1).replace('.', '-');
  return `${String(mins).padStart(2, '0')}m${rem}s`;
}

function uid(): string {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

function isCanvasMostlyBlank(canvas: HTMLCanvasElement): boolean {
  try {
    const probe = document.createElement('canvas');
    probe.width = 32;
    probe.height = 32;
    const pctx = probe.getContext('2d');
    if (!pctx) return false;
    pctx.drawImage(canvas, 0, 0, 32, 32);
    const data = pctx.getImageData(0, 0, 32, 32).data;
    let nonZero = 0;
    let sumLuma = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        const luma = data[i] + data[i + 1] + data[i + 2];
        sumLuma += luma;
        if (luma > 12) nonZero++;
      }
    }
    return nonZero < 4 && sumLuma < 200;
  } catch {
    return false;
  }
}

/** Fast Laplacian variance sharpness metric on a downsampled video frame (higher = crisper / less motion blur). */
function computeFrameSharpness(video: HTMLVideoElement): { score: number; blank: boolean } {
  const vw = video.videoWidth || 320;
  const vh = video.videoHeight || 180;
  const scale = Math.min(1, 240 / Math.max(1, vw, vh));
  const w = Math.max(24, Math.round(vw * scale));
  const h = Math.max(24, Math.round(vh * scale));

  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d');
  if (!ctx) return { score: 0, blank: false };
  ctx.drawImage(video, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  const gray = new Float32Array(w * h);
  let lumaSum = 0;
  let nonZero = 0;
  for (let i = 0; i < w * h; i++) {
    const g = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    gray[i] = g;
    lumaSum += g;
    if (g > 8) nonZero++;
  }
  const blank = nonZero < w * h * 0.02 && lumaSum / (w * h) < 4;
  if (blank) return { score: -1, blank: true };

  // Laplacian variance across interior pixels
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      const idx = row + x;
      const lap =
        4 * gray[idx] -
        gray[idx - 1] -
        gray[idx + 1] -
        gray[idx - w] -
        gray[idx + w];
      sum += lap;
      sumSq += lap * lap;
      count++;
    }
  }
  if (count === 0) return { score: 0, blank: false };
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return { score: variance, blank: false };
}

async function videoToPngBlob(
  video: HTMLVideoElement
): Promise<{ blob: Blob; width: number; height: number; blank: boolean }> {
  const width = video.videoWidth || 1280;
  const height = video.videoHeight || 720;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('No se pudo inicializar el lienzo de captura');
  ctx.drawImage(video, 0, 0, width, height);
  const blank = isCanvasMostlyBlank(canvas);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('No se pudo exportar el frame a PNG'))),
      'image/png'
    );
  });
  return { blob, width, height, blank };
}

function seekVideoFast(video: HTMLVideoElement, targetSec: number, timeoutMs = 4500): Promise<boolean> {
  return new Promise((resolve) => {
    const dur = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const clamped = Math.max(0, Math.min(targetSec, Math.max(0, dur - 0.005)));

    if (Math.abs(video.currentTime - clamped) < 0.015 && video.readyState >= 2) {
      window.requestAnimationFrame(() => resolve(true));
      return;
    }

    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      window.requestAnimationFrame(() => resolve(ok));
    };

    const onSeeked = () => finish(true);
    const onError = () => finish(false);
    const timer = window.setTimeout(() => finish(video.readyState >= 2), timeoutMs);

    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    try {
      video.currentTime = clamped;
    } catch {
      finish(false);
    }
  });
}

async function withHiddenVideo<T>(
  videoUrl: string,
  fn: (video: HTMLVideoElement, duration: number) => Promise<T>
): Promise<T> {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.setAttribute('playsinline', 'true');
  video.setAttribute('webkit-playsinline', 'true');
  video.preload = 'auto';

  video.style.position = 'fixed';
  video.style.left = '-9999px';
  video.style.top = '0';
  video.style.width = '2px';
  video.style.height = '2px';
  video.style.opacity = '0.001';
  video.style.pointerEvents = 'none';
  document.body.appendChild(video);

  try {
    video.src = videoUrl;
    video.load();

    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(
        () => reject(new Error('No se pudo cargar el vídeo para extraer los frames.')),
        12000
      );
      const done = () => {
        window.clearTimeout(timer);
        video.removeEventListener('loadeddata', done);
        video.removeEventListener('canplay', done);
        video.removeEventListener('error', onErr);
        resolve();
      };
      const onErr = () => {
        window.clearTimeout(timer);
        video.removeEventListener('loadeddata', done);
        video.removeEventListener('canplay', done);
        video.removeEventListener('error', onErr);
        reject(new Error('No se pudo decodificar este formato de vídeo en el navegador.'));
      };
      video.addEventListener('loadeddata', done);
      video.addEventListener('canplay', done);
      video.addEventListener('error', onErr);
      if (video.readyState >= 2) done();
    });

    try {
      await video.play();
      video.pause();
    } catch {
      // autoplay blocked or unnecessary
    }

    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    return await fn(video, duration);
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.remove();
  }
}

/** Captures the exact frame currently visible in a <video> element at full native resolution. */
export async function captureCurrentVideoFrame(
  video: HTMLVideoElement,
  baseFilename: string,
  kind: ExtractedFrame['kind'] = 'custom',
  customLabel?: string
): Promise<ExtractedFrame> {
  if (video.readyState < 2 || !video.videoWidth) {
    throw new Error('Espera un instante a que cargue el vídeo antes de capturar el frame.');
  }
  const ts = Math.max(0, video.currentTime || 0);
  const { blob, width, height } = await videoToPngBlob(video);
  const stem = stripExt(baseFilename);
  const suffix =
    kind === 'first'
      ? 'primer_frame'
      : kind === 'last'
        ? 'ultimo_frame'
        : kind === 'sharpest'
          ? 'frame_mas_nitido'
          : `frame_${formatStampForFile(ts)}`;

  return {
    id: uid(),
    kind,
    label:
      customLabel ||
      (kind === 'first'
        ? 'Primer frame'
        : kind === 'last'
          ? 'Último frame'
          : kind === 'sharpest'
            ? 'Frame más nítido'
            : 'Frame capturado'),
    timestampSec: ts,
    width,
    height,
    blob,
    url: URL.createObjectURL(blob),
    filename: `${stem}_${suffix}.png`,
  };
}

/** Extracts BOTH the first frame and the last frame of a video URL using an isolated hidden <video>. */
export async function extractFirstAndLastFrames(
  videoUrl: string,
  baseFilename: string
): Promise<{ first: ExtractedFrame; last: ExtractedFrame; durationSec: number }> {
  return withHiddenVideo(videoUrl, async (video, duration) => {
    const stem = stripExt(baseFilename);

    let firstShot: { blob: Blob; width: number; height: number; ts: number } | null = null;
    const firstCandidates = duration > 0.15 ? [0, 0.04, 0.12] : [0];
    for (const t of firstCandidates) {
      await seekVideoFast(video, t);
      const res = await videoToPngBlob(video);
      if (!firstShot || !res.blank) {
        firstShot = { blob: res.blob, width: res.width, height: res.height, ts: video.currentTime || t };
      }
      if (!res.blank) break;
    }
    if (!firstShot) {
      const res = await videoToPngBlob(video);
      firstShot = { blob: res.blob, width: res.width, height: res.height, ts: 0 };
    }

    let lastShot: { blob: Blob; width: number; height: number; ts: number } | null = null;
    const lastCandidates =
      duration > 0.4
        ? [
            Math.max(0, duration - 0.05),
            Math.max(0, duration - 0.14),
            Math.max(0, duration - 0.3),
            Math.max(0, duration - 0.6),
          ]
        : [Math.max(0, duration - 0.02)];

    for (const t of lastCandidates) {
      await seekVideoFast(video, t);
      const res = await videoToPngBlob(video);
      if (!lastShot || !res.blank) {
        lastShot = { blob: res.blob, width: res.width, height: res.height, ts: video.currentTime || t };
      }
      if (!res.blank) break;
    }
    if (!lastShot) {
      lastShot = { ...firstShot };
    }

    const first: ExtractedFrame = {
      id: uid(),
      kind: 'first',
      label: 'Primer frame',
      timestampSec: firstShot.ts,
      width: firstShot.width,
      height: firstShot.height,
      blob: firstShot.blob,
      url: URL.createObjectURL(firstShot.blob),
      filename: `${stem}_primer_frame.png`,
    };

    const last: ExtractedFrame = {
      id: uid(),
      kind: 'last',
      label: 'Último frame',
      timestampSec: lastShot.ts,
      width: lastShot.width,
      height: lastShot.height,
      blob: lastShot.blob,
      url: URL.createObjectURL(lastShot.blob),
      filename: `${stem}_ultimo_frame.png`,
    };

    return { first, last, durationSec: duration };
  });
}

/** Extracts an interactive visual filmstrip of evenly spaced thumbnails across the video timeline. */
export async function extractVideoFilmstrip(
  videoUrl: string,
  count = 10
): Promise<FilmstripThumb[]> {
  return withHiddenVideo(videoUrl, async (video, duration) => {
    if (duration <= 0.05) return [];
    const n = Math.max(4, Math.min(14, count));
    const thumbs: FilmstripThumb[] = [];
    const vw = video.videoWidth || 320;
    const vh = video.videoHeight || 180;
    const tw = 160;
    const th = Math.max(60, Math.round((vh / Math.max(1, vw)) * tw));
    const canvas = document.createElement('canvas');
    canvas.width = tw;
    canvas.height = th;
    const ctx = canvas.getContext('2d')!;

    for (let i = 0; i < n; i++) {
      const frac = n === 1 ? 0 : i / (n - 1);
      const t = Math.min(Math.max(0, duration - 0.06), frac * Math.max(0, duration - 0.06));
      await seekVideoFast(video, t, 2500);
      ctx.drawImage(video, 0, 0, tw, th);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((b) => resolve(b), 'image/jpeg', 0.72)
      );
      if (blob) {
        thumbs.push({
          timestampSec: video.currentTime || t,
          url: URL.createObjectURL(blob),
        });
      }
    }
    return thumbs;
  });
}

/** Scans evenly spaced frames across the video and extracts the single sharpest (least motion-blurred) frame. */
export async function findSharpestVideoFrame(
  videoUrl: string,
  baseFilename: string,
  samples = 14
): Promise<ExtractedFrame> {
  return withHiddenVideo(videoUrl, async (video, duration) => {
    const n = duration > 1 ? Math.max(6, Math.min(18, samples)) : 4;
    let bestTime = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < n; i++) {
      const frac = n === 1 ? 0 : i / (n - 1);
      const t = Math.min(Math.max(0, duration - 0.06), frac * Math.max(0, duration - 0.06));
      await seekVideoFast(video, t, 2500);
      const { score, blank } = computeFrameSharpness(video);
      if (!blank && score > bestScore) {
        bestScore = score;
        bestTime = video.currentTime || t;
      }
    }

    await seekVideoFast(video, bestTime, 3500);
    const { blob, width, height } = await videoToPngBlob(video);
    const stem = stripExt(baseFilename);
    return {
      id: uid(),
      kind: 'sharpest',
      label: 'Mejor frame (más nítido)',
      timestampSec: bestTime,
      width,
      height,
      blob,
      url: URL.createObjectURL(blob),
      filename: `${stem}_mas_nitido_${formatStampForFile(bestTime)}.png`,
    };
  });
}

/** Converts a frame Blob to chosen format (PNG/JPG/WebP) and optional aspect-ratio center crop (16:9, 9:16, 1:1). */
export async function convertFrameBlob(
  blob: Blob,
  format: FrameFormat,
  aspect: FrameAspect,
  filename: string
): Promise<{ blob: Blob; filename: string; width: number; height: number }> {
  const ext = format === 'jpeg' ? 'jpg' : format;
  const aspectTag = aspect === 'original' ? '' : `_${aspect.replace(':', 'x')}`;
  const outName = `${filename.replace(/\.[^.]+$/, '')}${aspectTag}.${ext}`;

  if (format === 'png' && aspect === 'original' && blob.type === 'image/png') {
    return { blob, filename: outName, width: 0, height: 0 };
  }

  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('No se pudo procesar el frame'));
      el.src = url;
    });

    const sw = img.naturalWidth || img.width;
    const sh = img.naturalHeight || img.height;
    let sx = 0;
    let sy = 0;
    let cw = sw;
    let ch = sh;

    if (aspect !== 'original') {
      const targetRatio = aspect === '16:9' ? 16 / 9 : aspect === '9:16' ? 9 / 16 : 1;
      const srcRatio = sw / Math.max(1, sh);
      if (srcRatio > targetRatio) {
        cw = Math.max(1, Math.round(sh * targetRatio));
        sx = Math.round((sw - cw) / 2);
      } else {
        ch = Math.max(1, Math.round(sw / targetRatio));
        sy = Math.round((sh - ch) / 2);
      }
    }

    const canvas = document.createElement('canvas');
    canvas.width = cw;
    canvas.height = ch;
    const ctx = canvas.getContext('2d')!;
    if (format === 'jpeg') {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, cw, ch);
    }
    ctx.drawImage(img, sx, sy, cw, ch, 0, 0, cw, ch);

    const mime =
      format === 'jpeg' ? 'image/jpeg' : format === 'webp' ? 'image/webp' : 'image/png';
    const quality = format === 'png' ? undefined : 0.92;
    const outBlob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Error convirtiendo frame'))), mime, quality)
    );
    return { blob: outBlob, filename: outName, width: cw, height: ch };
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);
}

/** Packages multiple extracted frames into a single ZIP file for instant download. */
export async function exportFramesZip(
  items: Array<{ filename: string; blob: Blob }>,
  zipName = 'FriendFinder_Frames.zip'
): Promise<void> {
  const zip = new JSZip();
  const used = new Set<string>();
  for (const item of items) {
    let name = item.filename;
    let idx = 2;
    while (used.has(name)) {
      name = item.filename.replace(/(\.[^.]+)$/, `_${idx++}$1`);
    }
    used.add(name);
    zip.file(name, item.blob);
  }
  const outBlob = await zip.generateAsync({ type: 'blob' });
  downloadBlob(outBlob, zipName);
}
