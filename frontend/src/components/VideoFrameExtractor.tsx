import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Check,
  ChevronLeft,
  ChevronRight,
  Columns2,
  Copy,
  Download,
  RefreshCw,
  Scissors,
  Search,
  SkipBack,
  SkipForward,
  Sparkles,
  UserPlus,
  X,
} from 'lucide-react';
import { formatTime } from '../api';
import { removeBackgroundFromBlob } from '../browserEngine/ml';
import { copyImageBlobToClipboard } from '../clipboard';
import {
  type ExtractedFrame,
  type FilmstripThumb,
  type FrameAspect,
  type FrameFormat,
  captureCurrentVideoFrame,
  convertFrameBlob,
  downloadBlob,
  extractFirstAndLastFrames,
  extractVideoFilmstrip,
  findSharpestVideoFrame,
} from '../videoFrames';

interface Props {
  videoUrl: string;
  filename: string;
  videoElementRef?: React.RefObject<HTMLVideoElement | null>;
  compact?: boolean;
  onSearchFrame?: (file: File) => void;
  onAddFrameToPeople?: (file: File) => void;
  onToast: (msg: string) => void;
}

export function VideoFrameExtractor({
  videoUrl,
  filename,
  videoElementRef,
  compact = false,
  onSearchFrame,
  onAddFrameToPeople,
  onToast,
}: Props) {
  const [firstFrame, setFirstFrame] = useState<ExtractedFrame | null>(null);
  const [lastFrame, setLastFrame] = useState<ExtractedFrame | null>(null);
  const [customFrames, setCustomFrames] = useState<ExtractedFrame[]>([]);
  const [filmstrip, setFilmstrip] = useState<FilmstripThumb[]>([]);
  const [extracting, setExtracting] = useState(false);
  const [findingSharpest, setFindingSharpest] = useState(false);
  const [cuttingId, setCuttingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Output options & Before/After comparator
  const [format, setFormat] = useState<FrameFormat>('png');
  const [aspect, setAspect] = useState<FrameAspect>('original');
  const [showCompare, setShowCompare] = useState(false);
  const [compareSplit, setCompareSplit] = useState(50);

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

      // Generate interactive filmstrip thumbnails asynchronously right after 1st/last so UI feels instant
      void extractVideoFilmstrip(videoUrl, compact ? 8 : 10)
        .then((thumbs) => {
          thumbs.forEach((t) => trackUrl(t.url));
          setFilmstrip(thumbs);
        })
        .catch(() => {
          // optional filmstrip
        });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'No se pudieron extraer los frames automáticamente.'
      );
    } finally {
      setExtracting(false);
    }
  }, [videoUrl, filename, compact]);

  useEffect(() => {
    setFirstFrame(null);
    setLastFrame(null);
    setCustomFrames([]);
    setFilmstrip([]);
    setShowCompare(false);
    setError(null);
    void runExtract();
    return () => {
      revokeAll();
      if (copiedTimerRef.current) window.clearTimeout(copiedTimerRef.current);
    };
  }, [runExtract, revokeAll]);

  const handleCopy = useCallback(
    async (frame: ExtractedFrame) => {
      try {
        let blobToCopy = frame.blob;
        if (aspect !== 'original') {
          const converted = await convertFrameBlob(frame.blob, 'png', aspect, frame.filename);
          blobToCopy = converted.blob;
        }
        await copyImageBlobToClipboard(blobToCopy);
        setCopiedId(frame.id);
        if (copiedTimerRef.current) window.clearTimeout(copiedTimerRef.current);
        copiedTimerRef.current = window.setTimeout(() => setCopiedId(null), 2200);
        const aspectHint = aspect === 'original' ? '' : ` (${aspect})`;
        onToast(`${frame.label}${aspectHint} copiado al portapapeles. Pega con Ctrl+V / ⌘V.`);
      } catch (err) {
        onToast(err instanceof Error ? err.message : 'No se pudo copiar al portapapeles');
      }
    },
    [aspect, onToast]
  );

  const handleDownload = async (frame: ExtractedFrame) => {
    try {
      const targetFormat: FrameFormat = frame.isCutout ? 'png' : format;
      const converted = await convertFrameBlob(frame.blob, targetFormat, aspect, frame.filename);
      downloadBlob(converted.blob, converted.filename);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error descargando el frame');
    }
  };

  const handleCaptureCurrent = useCallback(
    async (alsoCopy = false) => {
      const video = videoElementRef?.current;
      if (!video) return;
      try {
        const frame = await captureCurrentVideoFrame(video, filename, 'custom');
        trackUrl(frame.url);
        setCustomFrames((prev) => [frame, ...prev]);
        if (alsoCopy) {
          await handleCopy(frame);
        } else {
          onToast(`Frame de ${formatTime(frame.timestampSec)} capturado listo para copiar o descargar.`);
        }
      } catch (err) {
        onToast(err instanceof Error ? err.message : 'No se pudo capturar el frame actual');
      }
    },
    [videoElementRef, filename, handleCopy, onToast]
  );

  const handleFindSharpest = async () => {
    if (!videoUrl || findingSharpest) return;
    setFindingSharpest(true);
    try {
      const sharp = await findSharpestVideoFrame(videoUrl, filename, 14);
      trackUrl(sharp.url);
      setCustomFrames((prev) => [sharp, ...prev]);
      if (videoElementRef?.current) {
        videoElementRef.current.pause();
        videoElementRef.current.currentTime = sharp.timestampSec;
      }
      onToast(`Frame más nítido detectado en ${formatTime(sharp.timestampSec)}.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo analizar la nitidez del vídeo');
    } finally {
      setFindingSharpest(false);
    }
  };

  const handleRemoveBg = async (frame: ExtractedFrame) => {
    if (cuttingId) return;
    setCuttingId(frame.id);
    try {
      const cutoutBlob = await removeBackgroundFromBlob(frame.blob);
      const url = URL.createObjectURL(cutoutBlob);
      trackUrl(url);
      const cutoutFrame: ExtractedFrame = {
        id: `cut_${Date.now().toString(36)}`,
        kind: 'cutout',
        label: `${frame.label} (sin fondo)`,
        timestampSec: frame.timestampSec,
        width: frame.width,
        height: frame.height,
        blob: cutoutBlob,
        url,
        filename: frame.filename.replace(/\.[^.]+$/, '_sin_fondo.png'),
        isCutout: true,
      };
      setCustomFrames((prev) => [cutoutFrame, ...prev]);
      onToast('Fondo eliminado en PNG transparente. Pulsa Copiar para pegarlo.');
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo quitar el fondo de este frame');
    } finally {
      setCuttingId(null);
    }
  };

  const makeFileFromFrame = async (frame: ExtractedFrame): Promise<File> => {
    const converted = await convertFrameBlob(frame.blob, 'png', aspect, frame.filename);
    return new File([converted.blob], converted.filename, { type: 'image/png' });
  };

  const stepVideo = useCallback(
    (deltaSec: number) => {
      const video = videoElementRef?.current;
      if (!video || !Number.isFinite(video.duration)) return;
      video.pause();
      video.currentTime = Math.max(0, Math.min(video.duration - 0.01, video.currentTime + deltaSec));
    },
    [videoElementRef]
  );

  const jumpVideoTo = useCallback(
    (sec: number) => {
      const video = videoElementRef?.current;
      if (!video || !Number.isFinite(video.duration)) return;
      video.pause();
      video.currentTime = Math.max(0, Math.min(video.duration - 0.02, sec));
    },
    [videoElementRef]
  );

  const jumpVideoEnd = (toEnd: boolean) => {
    const video = videoElementRef?.current;
    if (!video || !Number.isFinite(video.duration)) return;
    video.pause();
    video.currentTime = toEnd ? Math.max(0, video.duration - 0.05) : 0;
  };

  // Keyboard shortcuts: Left/Right = -1/+1 frame, 1 = copy 1st, 2 = copy last, C = capture & copy current
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.key === 'ArrowLeft' && videoElementRef?.current) {
        e.preventDefault();
        stepVideo(-1 / 30);
      } else if (e.key === 'ArrowRight' && videoElementRef?.current) {
        e.preventDefault();
        stepVideo(1 / 30);
      } else if (e.key === '1' && firstFrame) {
        e.preventDefault();
        void handleCopy(firstFrame);
      } else if (e.key === '2' && lastFrame) {
        e.preventDefault();
        void handleCopy(lastFrame);
      } else if ((e.key === 'c' || e.key === 'C') && videoElementRef?.current) {
        e.preventDefault();
        void handleCaptureCurrent(true);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [firstFrame, lastFrame, videoElementRef, stepVideo, handleCopy, handleCaptureCurrent]);

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
    const isCutting = cuttingId === frame.id;
    const formatLabel = frame.isCutout ? 'PNG' : format === 'jpeg' ? 'JPG' : format.toUpperCase();

    return (
      <div key={frame.id} className="frame-card">
        <div
          className={`frame-card-preview ${frame.isCutout ? 'checkerboard-bg' : ''}`}
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
              {frame.width}×{frame.height} {aspect !== 'original' ? `· ${aspect}` : ''}
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
            onClick={() => void handleDownload(frame)}
            title={`Descargar en ${formatLabel}`}
          >
            <Download size={14} /> {formatLabel}
          </button>
        </div>

        <div className="frame-card-ai-bar">
          {!frame.isCutout && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              style={{ fontSize: '0.74rem', padding: '4px 7px' }}
              disabled={Boolean(cuttingId)}
              onClick={() => void handleRemoveBg(frame)}
              title="Recortar persona sin fondo en PNG transparente"
            >
              <Scissors size={12} /> {isCutting ? 'Recortando...' : 'Sin fondo'}
            </button>
          )}
          {onSearchFrame && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              style={{ fontSize: '0.74rem', padding: '4px 7px' }}
              onClick={async () => {
                const file = await makeFileFromFrame(frame);
                onSearchFrame(file);
              }}
              title="Buscar quién aparece en este frame en toda tu biblioteca"
            >
              <Search size={12} /> Buscar cara
            </button>
          )}
          {onAddFrameToPeople && (
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              style={{ fontSize: '0.74rem', padding: '4px 7px' }}
              onClick={async () => {
                const file = await makeFileFromFrame(frame);
                onAddFrameToPeople(file);
              }}
              title="Enviar este frame a Personas para detectarlo y guardarlo"
            >
              <UserPlus size={12} /> A Personas
            </button>
          )}
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
            Atajos: <kbd>←</kbd>/<kbd>→</kbd> 1 frame · <kbd>1</kbd> copiar 1.º · <kbd>2</kbd>{' '}
            copiar último · <kbd>C</kbd> capturar y copiar actual
          </div>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
          {videoElementRef && (
            <button
              type="button"
              id="btn-capture-current-frame"
              className="btn btn-dark btn-sm"
              onClick={() => void handleCaptureCurrent(false)}
              title="Captura el fotograma exacto que estás viendo ahora (tecla C para copiar al instante)"
            >
              <Camera size={14} /> Capturar actual
            </button>
          )}
          <button
            type="button"
            className="btn btn-sm"
            disabled={findingSharpest || extracting}
            onClick={() => void handleFindSharpest()}
            title="Analiza el vídeo automáticamente para encontrar el fotograma menos borroso"
          >
            <Sparkles size={14} style={{ color: 'var(--accent)' }} />
            {findingSharpest ? 'Buscando...' : 'Más nítido'}
          </button>
          {firstFrame && lastFrame && (
            <button
              type="button"
              className={`btn btn-sm ${showCompare ? 'btn-primary' : ''}`}
              onClick={() => setShowCompare((v) => !v)}
              title="Comparar visualmente primer y último frame con deslizador"
            >
              <Columns2 size={14} /> Comparar
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
          </button>
        </div>
      </div>

      {/* Format, Aspect Ratio & Precision Step Bar */}
      <div className="frame-step-bar">
        {videoElementRef && (
          <>
            <button type="button" className="btn btn-sm" onClick={() => jumpVideoEnd(false)}>
              <SkipBack size={13} /> Inicio
            </button>
            <button type="button" className="btn btn-sm" onClick={() => stepVideo(-0.5)}>
              −0.5s
            </button>
            <button type="button" className="btn btn-sm" onClick={() => stepVideo(-1 / 30)}>
              <ChevronLeft size={13} /> 1f
            </button>
            <button type="button" className="btn btn-sm" onClick={() => stepVideo(1 / 30)}>
              1f <ChevronRight size={13} />
            </button>
            <button type="button" className="btn btn-sm" onClick={() => stepVideo(0.5)}>
              +0.5s
            </button>
            <button type="button" className="btn btn-sm" onClick={() => jumpVideoEnd(true)}>
              Final <SkipForward size={13} />
            </button>
          </>
        )}

        <div
          style={{
            marginLeft: 'auto',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            flexWrap: 'wrap',
          }}
        >
          <select
            aria-label="Formato de descarga"
            className="select"
            style={{ width: 'auto', padding: '4px 8px', fontSize: '0.78rem' }}
            value={format}
            onChange={(e) => setFormat(e.target.value as FrameFormat)}
          >
            <option value="png">PNG (calidad máxima)</option>
            <option value="jpeg">JPG (ligero)</option>
            <option value="webp">WebP (web)</option>
          </select>
          <select
            aria-label="Relación de aspecto"
            className="select"
            style={{ width: 'auto', padding: '4px 8px', fontSize: '0.78rem' }}
            value={aspect}
            onChange={(e) => setAspect(e.target.value as FrameAspect)}
          >
            <option value="original">Ratio original</option>
            <option value="16:9">Recorte 16:9</option>
            <option value="9:16">Recorte 9:16</option>
            <option value="1:1">Recorte 1:1</option>
          </select>
        </div>
      </div>

      {/* Interactive Filmstrip Scrubber */}
      {filmstrip.length > 0 && videoElementRef && (
        <div className="filmstrip-strip" aria-label="Tira de fotogramas del vídeo">
          {filmstrip.map((thumb, idx) => (
            <button
              key={idx}
              type="button"
              className="filmstrip-thumb"
              onClick={() => jumpVideoTo(thumb.timestampSec)}
              title={`Ir a ${formatTime(thumb.timestampSec)}`}
            >
              <img src={thumb.url} alt={`Frame ${formatTime(thumb.timestampSec)}`} loading="lazy" />
              <span className="filmstrip-ts mono">{formatTime(thumb.timestampSec)}</span>
            </button>
          ))}
        </div>
      )}

      {/* Before / After Comparator (First vs Last Frame) */}
      {showCompare && firstFrame && lastFrame && (
        <div className="compare-box">
          <div className="compare-stage">
            <img src={lastFrame.url} alt="Último frame" className="compare-img-base" />
            <div
              className="compare-img-overlay"
              style={{ clipPath: `inset(0 ${100 - compareSplit}% 0 0)` }}
            >
              <img src={firstFrame.url} alt="Primer frame" />
            </div>
            <div className="compare-divider" style={{ left: `${compareSplit}%` }} />
            <span className="badge badge-dark compare-label-left">
              1.º ({formatTime(firstFrame.timestampSec)})
            </span>
            <span className="badge badge-dark compare-label-right">
              Último ({formatTime(lastFrame.timestampSec)})
            </span>
            <input
              type="range"
              min={0}
              max={100}
              value={compareSplit}
              aria-label="Comparar primer y último frame"
              className="compare-range"
              onChange={(e) => setCompareSplit(Number(e.target.value))}
            />
          </div>
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
            Capturas extra y recortes sin fondo ({customFrames.length}):
          </div>
          <div className="frame-cards-grid">
            {customFrames.map((f) => renderFrameCard(f, true))}
          </div>
        </div>
      )}
    </div>
  );
}
