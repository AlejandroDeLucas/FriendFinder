/** Helpers for pasting images via Ctrl/Cmd+V (paste event) or via Clipboard API button (mobile & desktop). */

function makePastedFilename(mime: string, index = 0): string {
  const ext = mime.includes('png')
    ? 'png'
    : mime.includes('webp')
      ? 'webp'
      : mime.includes('gif')
        ? 'gif'
        : 'jpg';
  const now = new Date();
  const stamp = [
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join('');
  const suffix = index > 0 ? `_${index + 1}` : '';
  return `portapapeles_${stamp}${suffix}.${ext}`;
}

export function extractImagesFromPasteEvent(event: ClipboardEvent): File[] {
  const files: File[] = [];
  const items = event.clipboardData?.items;
  if (items) {
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const raw = item.getAsFile();
        if (raw) {
          const name =
            raw.name && raw.name !== 'image.png'
              ? raw.name
              : makePastedFilename(raw.type || item.type, files.length);
          files.push(new File([raw], name, { type: raw.type || item.type }));
        }
      }
    }
  }
  if (files.length === 0 && event.clipboardData?.files?.length) {
    Array.from(event.clipboardData.files).forEach((f) => {
      if (f.type.startsWith('image/')) {
        files.push(f);
      }
    });
  }
  return files;
}

export async function readImagesFromClipboardApi(): Promise<File[]> {
  if (!navigator.clipboard || typeof navigator.clipboard.read !== 'function') {
    throw new Error('Usa Ctrl+V / Cmd+V para pegar (este navegador no soporta el botón de portapapeles).');
  }
  const clipboardItems = await navigator.clipboard.read();
  const files: File[] = [];
  for (const item of clipboardItems) {
    const imgType = item.types.find((t) => t.startsWith('image/'));
    if (!imgType) continue;
    const blob = await item.getType(imgType);
    files.push(new File([blob], makePastedFilename(imgType, files.length), { type: imgType }));
  }
  if (files.length === 0) {
    throw new Error('No hay ninguna imagen en el portapapeles.');
  }
  return files;
}

export function extractVideosFromPasteEvent(event: ClipboardEvent): File[] {
  const files: File[] = [];
  const items = event.clipboardData?.items;
  const isVideoFile = (f: File) =>
    f.type.startsWith('video/') || /\.(mp4|mov|webm|mkv|m4v|avi|3gp|ogv)$/i.test(f.name);

  if (items) {
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.kind === 'file') {
        const raw = item.getAsFile();
        if (raw && isVideoFile(raw)) {
          files.push(raw);
        }
      }
    }
  }
  if (files.length === 0 && event.clipboardData?.files?.length) {
    Array.from(event.clipboardData.files).forEach((f) => {
      if (isVideoFile(f)) files.push(f);
    });
  }
  return files;
}

async function ensurePngBlob(blob: Blob): Promise<Blob> {
  if (blob.type === 'image/png') return blob;
  const url = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('No se pudo convertir la imagen a PNG'));
      el.src = url;
    });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    canvas.getContext('2d')!.drawImage(img, 0, 0);
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Error generando PNG'))), 'image/png')
    );
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** Copies an image Blob (as lossless PNG) directly to the system clipboard so the user can Ctrl+V / paste anywhere. */
export async function copyImageBlobToClipboard(blob: Blob): Promise<void> {
  if (
    typeof navigator === 'undefined' ||
    !navigator.clipboard ||
    typeof navigator.clipboard.write !== 'function' ||
    typeof ClipboardItem === 'undefined'
  ) {
    throw new Error(
      'Tu navegador no permite copiar imágenes directamente. Usa el botón Descargar PNG.'
    );
  }
  // Passing a Promise<Blob> to ClipboardItem keeps Safari/iOS user-gesture activation intact!
  const pngPromise = blob.type === 'image/png' ? Promise.resolve(blob) : ensurePngBlob(blob);
  try {
    const item = new ClipboardItem({ 'image/png': pngPromise });
    await navigator.clipboard.write([item]);
  } catch {
    // Fallback for browsers that require a resolved Blob inside ClipboardItem
    const pngBlob = await pngPromise;
    const item = new ClipboardItem({ 'image/png': pngBlob });
    await navigator.clipboard.write([item]);
  }
}

