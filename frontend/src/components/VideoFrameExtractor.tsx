import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  RefreshCw,
  SkipBack,
  SkipForward,
  X,
} from 'lucide-react';
import { formatTime } from '../api';
import { copyImageBlobToClipboard } from '../clipboard';
import {
  type ExtractedFrame,
  captureCurrentVideoFrame,
  downloadBlob,
  extractFirstAndLastFrames,
} from '../videoFrames';

interface Props {
  videoUrl: string;
  filename: string;
  videoElementRef?: React.RefObject<HTMLVideoElement | null>;
  compact?: boolean;
  onToast: (msg: string) => void;
}

export function VideoFrameExtractor({
  videoUrl,
  filename,
  videoElementRef,
  compact = false,
  onToast,
}: Props) {
  const [firstFrame, setFirstFrame] = useState<ExtractedFrame | null>(null);
  const [lastFrame, setLastFrame] = useState<ExtractedFrame | null>(null);
  const [customFrames, setCustomFrames] = useState<ExtractedFrame[]>([]);
  const [extracting, setExtracting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const urlsToRevokeRef = useRef<Set<string>>(new Set());
  const copiedTimerRef = useRef<number | null>(null);

  const trackUrl = (url: string) => {
    urlsToRevokeRef.current.add(url);
  };

  const revokeAll = useCallback(() => {
    for (const url of urlsToRevokeRef.current) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore
      }
    }
    urlsToRevokeRef.current.clear();
  }, []);

  const runExtract = useCallback(async () => {
    if (!videoUrl) return;
    setExtracting(true);
    setError(null);
    try {
      const { first, last } = await extractFirstAndLastFrames(videoUrl, filename);
      trackUrl(first.url);
      trackUrl(last.url);
      setFirstFrame(first);
      setLastFrame(last);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'No se pudieron extraer los frames automáticamente.'
      );
    } finally {
      setExtracting(false);
    }
  }, [videoUrl, filename]);

  useEffect(() => {
    setFirstFrame(null);
    setLastFrame(null);
    setCustomFrames([]);
    setError(null);
    void runExtract();
    return () => {
      revokeAll();
      if (copiedTimerRef.current) window.clearTimeout(copiedTimerRef.current);
    };
  }, [runExtract, revokeAll]);

  const handleCopy = async (frame: ExtractedFrame) => {
    try {
      await copyImageBlobToClipboard(frame.blob);
      setCopiedId(frame.id);
      if (copiedTimerRef.current) window.clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = window.setTimeout(() => setCopiedId(null), 2200);
      onToast(`${frame.label} copiado al portapapeles. Ya puedes pegarlo con Ctrl+V / ⌘V.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo copiar al portapapeles');
    }
  };

  const handleCaptureCurrent = async () => {
    const video = videoElementRef?.current;
    if (!video) return;
    try {
      const frame = await captureCurrentVideoFrame(video, filename, 'custom');
      trackUrl(frame.url);
      setCustomFrames((prev) => [frame, ...prev]);
      onToast(`Frame de ${formatTime(frame.timestampSec)} capturado listo para copiar o descargar.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo capturar el frame actual');
    }
  };

  const stepVideo = (deltaSec: number) => {
    const video = videoElementRef?.current;
    if (!video || !Number.isFinite(video.duration)) return;
    video.pause();
    video.currentTime = Math.max(0, Math.min(video.duration - 0.01, video.currentTime + deltaSec));
  };

  const jumpVideo = (toEnd: boolean) => {
    const video = videoElementRef?.current;
    if (!video || !Number.isFinite(video.duration)) return;
    video.pause();
    video.currentTime = toEnd ? Math.max(0, video.duration - 0.05) : 0;
  };

  const removeCustomFrame = (id: string) => {
    setCustomFrames((prev) => {
      const target = prev.find((f) => f.id === id);
      if (target) {
        URL.revokeObjectURL(target.url);
        urlsToRevokeRef.current.delete(target.url);
      }
      return prev.filter((f) => f.id !== id);
    });
  };

  const renderFrameCard = (frame: ExtractedFrame, removable = false) => {
    const isCopied = copiedId === frame.id;
    return (
      <div key={frame.id} className="frame-card">
        <div
          className="frame-card-preview"
          onClick={() => void handleCopy(frame)}
          title="Clic para copiar directamente al portapapeles"
        >
          <img src={frame.url} alt={frame.label} />
          <div className="card-overlay-top">
            <span className="badge badge-dark" style={{ fontWeight: 700 }}>
              {frame.label}
            </span>
            <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
              <span className="badge badge-dark mono">{formatTime(frame.timestampSec)}</span>
              {removable && (
                <button
                  type="button"
                  aria-label="Quitar captura"
                  className="frame-remove-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeCustomFrame(frame.id);
                  }}
                >
                  <X size={13} />
                </button>
              )}
            </div>
          </div>
          <div className="card-overlay-bottom">
            <span className="badge badge-dark mono">
              {frame.width}×{frame.height} PNG
            </span>
          </div>
        </div>

        <div className="frame-card-actions">
          <button
            type="button"
            className={`btn btn-sm ${isCopied ? 'btn-success' : 'btn-primary'}`}
            style={{ flex: 1 }}
            onClick={() => void handleCopy(frame)}
          >
            {isCopied ? <Check size={14} /> : <Copy size={14} />}
            {isCopied ? '¡Copiado!' : 'Copiar'}
          </button>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => downloadBlob(frame.blob, frame.filename)}
            title={`Descargar ${frame.filename}`}
          >
            <Download size={14} /> PNG
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className={`frame-extractor ${compact ? 'compact' : ''}`}>
      <div className="frame-extractor-header">
        <div>
          <h3 style={{ margin: 0, fontSize: compact ? '0.95rem' : '1.05rem', fontWeight: 700 }}>
            Extracción rápida de frames (1.º y último)
          </h3>
          <div style={{ fontSize: '0.79rem', color: 'var(--ink-secondary)' }}>
            Pulsa <strong>Copiar</strong> o toca cualquier imagen para copiarla al portapapeles y
            pegarla con <kbd>Ctrl+V</kbd> / <kbd>⌘V</kbd>.
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          {videoElementRef && (
            <button
              type="button"
              id="btn-capture-current-frame"
              className="btn btn-dark btn-sm"
              onClick={() => void handleCaptureCurrent()}
              title="Captura el fotograma exacto que estás viendo ahora en el reproductor"
            >
              <Camera size={14} /> Capturar frame actual
            </button>
          )}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={extracting}
            onClick={() => void runExtract()}
            title="Volver a extraer primer y último frame"
          >
            <RefreshCw size={14} className={extracting ? 'spin-icon' : ''} />
            {extracting ? 'Extrayendo...' : 'Re-extraer'}
          </button>
        </div>
      </div>

      {videoElementRef && (
        <div className="frame-step-bar">
          <span style={{ fontSize: '0.77rem', color: 'var(--ink-muted)', fontWeight: 600 }}>
            Ajuste fino:
          </span>
          <button type="button" className="btn btn-sm" onClick={() => jumpVideo(false)}>
            <SkipBack size={13} /> Inicio
          </button>
          <button type="button" className="btn btn-sm" onClick={() => stepVideo(-0.5)}>
            −0.5s
          </button>
          <button type="button" className="btn btn-sm" onClick={() => stepVideo(-1 / 30)}>
            <ChevronLeft size={13} /> 1 frame
          </button>
          <button type="button" className="btn btn-sm" onClick={() => stepVideo(1 / 30)}>
            1 frame <ChevronRight size={13} />
          </button>
          <button type="button" className="btn btn-sm" onClick={() => stepVideo(0.5)}>
            +0.5s
          </button>
          <button type="button" className="btn btn-sm" onClick={() => jumpVideo(true)}>
            Final <SkipForward size={13} />
          </button>
        </div>
      )}

      {error && (
        <div
          style={{
            padding: '10px 14px',
            borderRadius: 'var(--radius-sm)',
            background: 'hsla(4, 70%, 50%, 0.08)',
            color: 'var(--danger)',
            fontSize: '0.83rem',
          }}
        >
          {error}
        </div>
      )}

      {extracting && !firstFrame && !lastFrame ? (
        <div className="frame-cards-grid">
          <div className="frame-card-skeleton">Extrayendo primer frame...</div>
          <div className="frame-card-skeleton">Extrayendo último frame...</div>
        </div>
      ) : (
        <div className="frame-cards-grid">
          {firstFrame && renderFrameCard(firstFrame)}
          {lastFrame && renderFrameCard(lastFrame)}
        </div>
      )}

      {customFrames.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div
            style={{
              fontSize: '0.82rem',
              fontWeight: 700,
              color: 'var(--ink-secondary)',
              marginBottom: 8,
            }}
          >
            Frames capturados del reproductor ({customFrames.length}):
          </div>
          <div className="frame-cards-grid">
            {customFrames.map((f) => renderFrameCard(f, true))}
          </div>
        </div>
      )}
    </div>
  );
}
