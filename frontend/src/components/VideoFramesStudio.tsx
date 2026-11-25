import { useEffect, useRef, useState } from 'react';
import { Clapperboard, Film, Play, Upload, X } from 'lucide-react';
import { assetUrl, formatBytes, formatTime } from '../api';
import { extractVideosFromPasteEvent } from '../clipboard';
import type { MediaItem } from '../types';
import { VideoFrameExtractor } from './VideoFrameExtractor';

interface Props {
  media: MediaItem[];
  onToast: (msg: string) => void;
}

interface ActiveVideoSource {
  id: string;
  url: string;
  filename: string;
  sizeBytes?: number;
  isLocalBlob: boolean;
}

export function VideoFramesStudio({ media, onToast }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [dragging, setDragging] = useState(false);
  const [activeVideo, setActiveVideo] = useState<ActiveVideoSource | null>(null);

  const libraryVideos = media.filter((m) => m.media_type === 'video' && !m.source_deleted);

  const selectLocalFile = (file: File) => {
    const isVideo =
      file.type.startsWith('video/') ||
      /\.(mp4|mov|webm|mkv|m4v|avi|3gp|ogv)$/i.test(file.name);
    if (!isVideo) {
      onToast('Por favor selecciona o pega un archivo de vídeo (MP4, MOV, WebM...).');
      return;
    }
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

  const clearActive = () => {
    setActiveVideo((prev) => {
      if (prev?.isLocalBlob) URL.revokeObjectURL(prev.url);
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
        selectLocalFile(videos[0]);
        onToast(`Vídeo "${videos[0].name}" pegado. Extrayendo primer y último frame...`);
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, []);

  // Revoke local object URL on unmount
  useEffect(() => {
    return () => {
      if (activeVideo?.isLocalBlob) {
        URL.revokeObjectURL(activeVideo.url);
      }
    };
  }, [activeVideo]);

  return (
    <section>
      <header className="page-header">
        <div>
          <h1 className="page-title">Extractor de frames de vídeo</h1>
          <p className="page-subtitle">
            Suelta, elige o pega (<kbd>Ctrl+V</kbd> / <kbd>⌘V</kbd>) cualquier vídeo, o toca uno de
            tu biblioteca para sacar al instante su <strong>primer y último frame</strong> listos
            para copiar y pegar en calidad original PNG.
          </p>
        </div>
        {activeVideo && (
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={() => inputRef.current?.click()}>
              <Upload size={15} /> Cambiar vídeo
            </button>
            <button className="btn btn-ghost" onClick={clearActive}>
              <X size={15} /> Cerrar
            </button>
          </div>
        )}
      </header>

      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) selectLocalFile(f);
          e.target.value = '';
        }}
      />

      {!activeVideo ? (
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
              const f = e.dataTransfer.files?.[0];
              if (f) selectLocalFile(f);
            }}
          >
            <div className="empty-studio-icon" style={{ marginBottom: 10 }}>
              <Clapperboard size={24} />
            </div>
            <div className="dropzone-title">Elige, arrastra o pega un vídeo</div>
            <p style={{ margin: 0, color: 'var(--ink-secondary)', fontSize: '0.88rem' }}>
              MP4, MOV, WebM, MKV · 100% instantáneo en tu navegador sin subir a ningún servidor
            </p>
          </div>
        </div>
      ) : (
        <div className="panel" style={{ marginBottom: 20, display: 'flex', flexDirection: 'column', gap: 18 }}>
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
            onToast={onToast}
          />
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
