/**
 * Fast, full-resolution client-side Video Frame Extractor.
 * Extracts exact First frame, Last frame (with EOS blank-frame fallback), and Current frame as lossless PNG.
 */

export interface ExtractedFrame {
  id: string;
  kind: 'first' | 'last' | 'custom';
  label: string;
  timestampSec: number;
  width: number;
  height: number;
  blob: Blob;
  url: string;
  filename: string;
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
      // Allow hardware video decoder 1 animation frame to paint the decoded frame cleanly
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

/** Captures the exact frame currently visible in a <video> element at full native resolution. */
export async function captureCurrentVideoFrame(
  video: HTMLVideoElement,
  baseFilename: string,
  kind: 'first' | 'last' | 'custom' = 'custom',
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
        : `frame_${formatStampForFile(ts)}`;

  return {
    id: uid(),
    kind,
    label:
      customLabel ||
      (kind === 'first' ? 'Primer frame' : kind === 'last' ? 'Último frame' : 'Frame capturado'),
    timestampSec: ts,
    width,
    height,
    blob,
    url: URL.createObjectURL(blob),
    filename: `${stem}_${suffix}.png`,
  };
}

/**
 * Extracts BOTH the first frame and the last frame of a video URL using an isolated hidden <video>
 * so it never disturbs playback or scrubber position on the visible player.
 */
export async function extractFirstAndLastFrames(
  videoUrl: string,
  baseFilename: string
): Promise<{ first: ExtractedFrame; last: ExtractedFrame; durationSec: number }> {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.setAttribute('playsinline', 'true');
  video.setAttribute('webkit-playsinline', 'true');
  video.preload = 'auto';

  // Keep mounted invisibly so iOS Safari / mobile Chrome hardware decoders paint frames cleanly
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

    // Warm up decoder gently on mobile Safari
    try {
      await video.play();
      video.pause();
    } catch {
      // autoplay blocked or unnecessary; seeking still works
    }

    const duration =
      Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const stem = stripExt(baseFilename);

    // 1. FIRST FRAME (try exact 0.0s, fallback to 0.04s if 0.0s decodes blank/empty)
    let firstShot: { blob: Blob; width: number; height: number; ts: number } | null = null;
    const firstCandidates = duration > 0.15 ? [0, 0.04, 0.12] : [0];
    for (const t of firstCandidates) {
      await seekVideoFast(video, t);
      const res = await videoToPngBlob(video);
      if (!firstShot || (!res.blank && firstShot)) {
        firstShot = { blob: res.blob, width: res.width, height: res.height, ts: video.currentTime || t };
      }
      if (!res.blank) break;
    }
    if (!firstShot) {
      const res = await videoToPngBlob(video);
      firstShot = { blob: res.blob, width: res.width, height: res.height, ts: 0 };
    }

    // 2. LAST FRAME (step just before EOS to avoid black end-of-stream frames in Chrome/Safari)
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
      if (!lastShot) {
        lastShot = { blob: res.blob, width: res.width, height: res.height, ts: video.currentTime || t };
      }
      if (!res.blank) {
        lastShot = { blob: res.blob, width: res.width, height: res.height, ts: video.currentTime || t };
        break;
      }
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
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.remove();
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
