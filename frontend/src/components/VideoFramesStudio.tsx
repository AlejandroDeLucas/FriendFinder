import { useEffect, useRef, useState } from 'react';
import {
  Check,
  Clapperboard,
  Copy,
  Download,
  Film,
  Layers,
  Play,
  Upload,
  X,
} from 'lucide-react';
import { assetUrl, formatBytes, formatTime } from '../api';
import { copyImageBlobToClipboard, extractVideosFromPasteEvent } from '../clipboard';
import type { MediaItem } from '../types';
import {
  type ExtractedFrame,
  exportFramesZip,
  extractFirstAndLastFrames,
} from '../videoFrames';
import { VideoFrameExtractor } from './VideoFrameExtractor';

interface Props {
  media: MediaItem[];
  onSearchFrame: (file: File) => void;
  onAddFrameToPeople: (file: File) => void;
  onToast: (msg: string) => void;
}

interface ActiveVideoSource {
  id: string;
  url: string;
  filename: string;
  sizeBytes?: number;
  isLocalBlob: boolean;
}

interface BatchVideoItem {
  id: string;
  url: string;
  filename: string;
  sizeBytes: number;
  status: 'pending' | 'done' | 'error';
  error?: string;
  first?: ExtractedFrame;
  last?: ExtractedFrame;
}

export function VideoFramesStudio({ media, onSearchFrame, onAddFrameToPeople, onToast }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [activeVideo, setActiveVideo] = useState<ActiveVideoSource | null>(null);
  const [batchItems, setBatchItems] = useState<BatchVideoItem[]>([]);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [exportingZip, setExportingZip] = useState(false);

  const batchUrlsRef = useRef<Set<string>>(new Set());
  const libraryVideos = media.filter((m) => m.media_type === 'video' && !m.source_deleted);

  const isVideoFile = (file: File) =>
    file.type.startsWith('video/') ||
    /\.(mp4|mov|webm|mkv|m4v|avi|3gp|ogv)$/i.test(file.name);

  const clearBatch = () => {
    for (const url of batchUrlsRef.current) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore
      }
    }
    batchUrlsRef.current.clear();
    setBatchItems([]);
  };

  const handleFilesSelected = async (incoming: FileList | File[]) => {
    const videos = Array.from(incoming).filter(isVideoFile);
    if (videos.length === 0) {
      onToast('Por favor selecciona, arrastra o pega archivos de vídeo (MP4, MOV, WebM...).');
      return;
    }

    if (videos.length === 1) {
      const file = videos[0];
      setActiveVideo((prev) => {
        if (prev?.isLocalBlob) URL.revokeObjectURL(prev.url);
        return {
          id: `local_${Date.now()}`,
          url: URL.createObjectURL(file),
          filename: file.name,
          sizeBytes: file.size,
          isLocalBlob: true,
        };
      });
      return;
    }

    // Multi-video batch mode! Extract 1st & Last frame of EVERY video sequentially
    clearBatch();
    setActiveVideo((prev) => {
      if (prev?.isLocalBlob) URL.revokeObjectURL(prev.url);
      return null;
    });

    const initial: BatchVideoItem[] = videos.map((f, idx) => {
      const url = URL.createObjectURL(f);
      batchUrlsRef.current.add(url);
      return {
        id: `batch_${Date.now()}_${idx}`,
        url,
        filename: f.name,
        sizeBytes: f.size,
        status: 'pending',
      };
    });
    setBatchItems(initial);

    for (const item of initial) {
      try {
        const { first, last } = await extractFirstAndLastFrames(item.url, item.filename);
        batchUrlsRef.current.add(first.url);
        batchUrlsRef.current.add(last.url);
        setBatchItems((prev) =>
          prev.map((x) =>
            x.id === item.id ? { ...x, status: 'done', first, last } : x
          )
        );
      } catch (err) {
        setBatchItems((prev) =>
          prev.map((x) =>
            x.id === item.id
              ? {
                  ...x,
                  status: 'error',
                  error: err instanceof Error ? err.message : 'Error extrayendo frames',
                }
              : x
          )
        );
      }
    }
    onToast(`¡Listo! 1.º y último frame extraídos de ${videos.length} vídeo(s).`);
  };

  const selectLibraryVideo = (item: MediaItem) => {
    const url = assetUrl.mediaStream(item.id);
    if (!url) {
      onToast('Este vídeo ya no tiene el original disponible.');
      return;
    }
    setActiveVideo((prev) => {
      if (prev?.isLocalBlob) URL.revokeObjectURL(prev.url);
      return {
        id: item.id,
        url,
        filename: item.original_filename,
        sizeBytes: item.file_size,
        isLocalBlob: false,
      };
    });
  };

  const openBatchVideoInEditor = (item: BatchVideoItem) => {
    setActiveVideo((prev) => {
      if (prev?.isLocalBlob && !batchUrlsRef.current.has(prev.url)) {
        URL.revokeObjectURL(prev.url);
      }
      return {
        id: item.id,
        url: item.url,
        filename: item.filename,
        sizeBytes: item.sizeBytes,
        isLocalBlob: false, // owned by batchUrlsRef
      };
    });
  };

  const handleCopyBatchFrame = async (key: string, frame: ExtractedFrame) => {
    try {
      await copyImageBlobToClipboard(frame.blob);
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((cur) => (cur === key ? null : cur)), 2000);
      onToast(`${frame.label} copiado al portapapeles.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo copiar');
    }
  };

  const handleDownloadBatchZip = async () => {
    const ready: Array<{ filename: string; blob: Blob }> = [];
    for (const item of batchItems) {
      if (item.first) ready.push({ filename: item.first.filename, blob: item.first.blob });
      if (item.last) ready.push({ filename: item.last.filename, blob: item.last.blob });
    }
    if (ready.length === 0) return;
    setExportingZip(true);
    try {
      await exportFramesZip(ready, 'FriendFinder_Frames_Lote.zip');
      onToast(`ZIP descargado con ${ready.length} frames PNG.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error creando ZIP');
    } finally {
      setExportingZip(false);
    }
  };

  const clearActive = () => {
    setActiveVideo((prev) => {
      if (prev?.isLocalBlob && !batchUrlsRef.current.has(prev.url)) {
        URL.revokeObjectURL(prev.url);
      }
      return null;
    });
  };

  // Support Ctrl+V / Cmd+V video paste directly on the Frames studio tab
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      const videos = extractVideosFromPasteEvent(e);
      if (videos.length > 0) {
        e.preventDefault();
        void handleFilesSelected(videos);
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, []);

  useEffect(() => {
    return () => {
      clearBatch();
    };
  }, []);

  const doneBatchCount = batchItems.filter((b) => b.status === 'done').length;

  return (
    <section>
      <header className="page-header">
        <div>
          <h1 className="page-title">Extractor de frames de vídeo</h1>
          <p className="page-subtitle">
            Suelta uno o varios vídeos a la vez (o pégalos con <kbd>Ctrl+V</kbd> / <kbd>⌘V</kbd>)
            para sacar al instante el <strong>1.º y último frame</strong>, buscar el más nítido,
            comparar antes/después o quitar el fondo en PNG.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {(activeVideo || batchItems.length > 0) && (
            <button className="btn" onClick={() => inputRef.current?.click()}>
              <Upload size={15} /> Elegir vídeo(s)
            </button>
          )}
          {activeVideo && (
            <button className="btn btn-ghost" onClick={clearActive}>
              <X size={15} /> Cerrar editor
            </button>
          )}
        </div>
      </header>

      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => {
          if (e.target.files && e.target.files.length > 0) {
            void handleFilesSelected(e.target.files);
          }
          e.target.value = '';
        }}
      />

      {!activeVideo && batchItems.length === 0 && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <div
            className={`dropzone ${dragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                void handleFilesSelected(e.dataTransfer.files);
              }
            }}
          >
            <div className="empty-studio-icon" style={{ marginBottom: 10 }}>
              <Clapperboard size={24} />
            </div>
            <div className="dropzone-title">Elige, arrastra o pega uno o varios vídeos</div>
            <p style={{ margin: 0, color: 'var(--ink-secondary)', fontSize: '0.88rem' }}>
              MP4, MOV, WebM, MKV · Suelta varios vídeos para sacar el 1.º y último frame en lote y
              descargarlos en ZIP
            </p>
          </div>
        </div>
      )}

      {activeVideo && (
        <div
          className="panel"
          style={{ marginBottom: 20, display: 'flex', flexDirection: 'column', gap: 18 }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 10,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
              <Film size={18} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <div style={{ minWidth: 0 }}>
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: '0.98rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {activeVideo.filename}
                </div>
                {activeVideo.sizeBytes ? (
                  <div style={{ fontSize: '0.78rem', color: 'var(--ink-muted)' }}>
                    {formatBytes(activeVideo.sizeBytes)}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          <div
            style={{
              background: 'var(--bg-darkroom)',
              borderRadius: 'var(--radius-md)',
              overflow: 'hidden',
              display: 'grid',
              placeItems: 'center',
            }}
          >
            <video
              key={activeVideo.id}
              ref={videoRef}
              src={activeVideo.url}
              crossOrigin="anonymous"
              controls
              playsInline
              style={{ width: '100%', maxHeight: 420 }}
            />
          </div>

          <VideoFrameExtractor
            videoUrl={activeVideo.url}
            filename={activeVideo.filename}
            videoElementRef={videoRef}
            onSearchFrame={onSearchFrame}
            onAddFrameToPeople={onAddFrameToPeople}
            onToast={onToast}
          />
        </div>
      )}

      {/* Multi-Video Batch Extraction Results */}
      {batchItems.length > 0 && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 12,
              marginBottom: 14,
            }}
          >
            <div>
              <h2 className="section-title" style={{ marginBottom: 2 }}>
                Lote de vídeos ({doneBatchCount}/{batchItems.length} procesados)
              </h2>
              <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-secondary)' }}>
                Copia al instante el 1.º o último frame de cada vídeo, ábrelo en el editor o descarga
                todo en un ZIP.
              </p>
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                className="btn btn-dark btn-sm"
                disabled={exportingZip || doneBatchCount === 0}
                onClick={() => void handleDownloadBatchZip()}
              >
                <Download size={14} />{' '}
                {exportingZip ? 'Empaquetando...' : `Descargar todos (${doneBatchCount * 2} PNG) ZIP`}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={clearBatch}>
                <X size={14} /> Limpiar lote
              </button>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {batchItems.map((item) => (
              <div key={item.id} className="batch-video-row">
                <div className="batch-video-meta">
                  <strong
                    style={{
                      fontSize: '0.92rem',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={item.filename}
                  >
                    {item.filename}
                  </strong>
                  <span style={{ fontSize: '0.76rem', color: 'var(--ink-muted)' }}>
                    {formatBytes(item.sizeBytes)}
                  </span>
                  <button
                    className="btn btn-sm"
                    style={{ marginTop: 6, width: 'fit-content' }}
                    onClick={() => openBatchVideoInEditor(item)}
                  >
                    <Layers size={13} /> Abrir en editor
                  </button>
                </div>

                {item.status === 'pending' ? (
                  <div className="batch-video-status">Extrayendo 1.º y último frame...</div>
                ) : item.status === 'error' ? (
                  <div className="batch-video-status" style={{ color: 'var(--danger)' }}>
                    {item.error || 'No se pudo leer este vídeo'}
                  </div>
                ) : (
                  <div className="batch-video-frames">
                    {[
                      { key: `${item.id}_first`, frame: item.first, btn: 'Copiar 1.º' },
                      { key: `${item.id}_last`, frame: item.last, btn: 'Copiar último' },
                    ].map(({ key, frame, btn }) =>
                      frame ? (
                        <div key={key} className="batch-mini-frame">
                          <img
                            src={frame.url}
                            alt={frame.label}
                            onClick={() => void handleCopyBatchFrame(key, frame)}
                            title="Clic para copiar"
                          />
                          <div className="batch-mini-actions">
                            <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                              {formatTime(frame.timestampSec)}
                            </span>
                            <button
                              className={`btn btn-sm ${copiedKey === key ? 'btn-success' : 'btn-primary'}`}
                              style={{ padding: '4px 8px', fontSize: '0.74rem' }}
                              onClick={() => void handleCopyBatchFrame(key, frame)}
                            >
                              {copiedKey === key ? <Check size={12} /> : <Copy size={12} />}
                              {copiedKey === key ? 'Copiado' : btn}
                            </button>
                          </div>
                        </div>
                      ) : null
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {libraryVideos.length > 0 && (
        <div className="panel">
          <div style={{ marginBottom: 12 }}>
            <h2 className="section-title" style={{ marginBottom: 2 }}>
              Vídeos de tu biblioteca ({libraryVideos.length})
            </h2>
            <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-secondary)' }}>
              Pulsa cualquier vídeo para extraer su primer y último frame en un clic.
            </p>
          </div>
          <div className="media-grid">
            {libraryVideos.map((item) => {
              const isCurrent = activeVideo?.id === item.id;
              return (
                <article
                  key={item.id}
                  className={`media-card ${isCurrent ? 'selected' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => selectLibraryVideo(item)}
                >
                  <div className="media-card-thumb">
                    {item.thumb_path ? (
                      <img
                        src={assetUrl.thumb(item.thumb_path)}
                        alt={item.original_filename}
                        loading="lazy"
                      />
                    ) : (
                      <div
                        style={{
                          width: '100%',
                          height: '100%',
                          display: 'grid',
                          placeItems: 'center',
                          color: '#aaa',
                        }}
                      >
                        <Film size={28} />
                      </div>
                    )}
                    <div className="card-overlay-top">
                      <span className="badge badge-dark">
                        <Play size={10} />{' '}
                        {item.duration_sec ? formatTime(item.duration_sec) : 'Vídeo'}
                      </span>
                      {isCurrent && <span className="badge badge-accent">Abierto</span>}
                    </div>
                  </div>
                  <div className="media-card-body">
                    <div
                      style={{
                        fontWeight: 700,
                        fontSize: '0.88rem',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {item.original_filename}
                    </div>
                    <div style={{ fontSize: '0.76rem', color: 'var(--ink-muted)' }}>
                      {formatBytes(item.file_size)}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
