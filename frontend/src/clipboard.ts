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
