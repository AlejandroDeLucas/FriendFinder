import { useRef, useState } from 'react';
import {
  CheckCircle2,
  Film,
  Image as ImageIcon,
  Layers,
  Sparkles,
  Upload,
  X,
  XCircle,
} from 'lucide-react';
import { api, formatBytes } from '../api';
import type { JobInfo } from '../types';

interface Props {
  activeJob: JobInfo | null;
  lastFinishedJob: JobInfo | null;
  defaultSampleFps: number;
  onJobStarted: (jobId: string) => void;
  onGoPeople: () => void;
  onToast: (msg: string) => void;
}

export function UploadStudio({
  activeJob,
  lastFinishedJob,
  defaultSampleFps,
  onJobStarted,
  onGoPeople,
  onToast,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [sampleFps, setSampleFps] = useState<number>(defaultSampleFps || 1.0);
  const [uploading, setUploading] = useState(false);

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => `${f.name}-${f.size}`));
      const unique = arr.filter((f) => !existing.has(`${f.name}-${f.size}`));
      return [...prev, ...unique];
    });
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleStart = async () => {
    if (files.length === 0) return;
    setUploading(true);
    try {
      const res = await api.uploadMedia(files, sampleFps);
      setFiles([]);
      onJobStarted(res.job_id);
      if (res.skipped.length > 0) {
        onToast(
          `Subidos ${res.uploaded_count} archivo(s). ${res.skipped.length} omitido(s) por formato.`
        );
      } else {
        onToast(`Analizando ${res.uploaded_count} archivo(s) en segundo plano...`);
      }
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al subir archivos');
    } finally {
      setUploading(false);
    }
  };

  const isVideoFile = (f: File) =>
    f.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <header className="page-header" style={{ marginBottom: 0 }}>
        <div>
          <h1 className="page-title">Subir fotos y vídeos</h1>
          <p className="page-subtitle">
            Arrastra una o varias fotos y vídeos. Se analizarán frame a frame en tu equipo para
            detectar a cada persona, extraer los mejores retratos y recortes PNG sin otras personas,
            y descartar automáticamente las tomas redundantes o de menor calidad.
          </p>
        </div>
      </header>

      <div className="panel">
        <div
          id="upload-dropzone"
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
            if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept="image/*,video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files?.length) addFiles(e.target.files);
              e.target.value = '';
            }}
          />
          <div className="empty-studio-icon" style={{ marginBottom: 10 }}>
            <Upload size={25} />
          </div>
          <div className="dropzone-title">Arrastra aquí tus fotos o vídeos</div>
          <p style={{ margin: 0, color: 'var(--ink-secondary)', fontSize: '0.9rem' }}>
            Soporta JPG, PNG, WebP, MP4, MOV, MKV, AVI, WebM · Todo se procesa 100% en local
          </p>
        </div>

        <div
          style={{
            marginTop: 18,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 14,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <label htmlFor="select-fps" style={{ fontSize: '0.86rem', fontWeight: 600 }}>
              Frecuencia en vídeos:
            </label>
            <select
              id="select-fps"
              className="select"
              style={{ width: 'auto' }}
              value={sampleFps}
              onChange={(e) => setSampleFps(Number(e.target.value))}
            >
              <option value={0.5}>0.5 frames/seg (muy rápido, vídeos largos)</option>
              <option value={1.0}>1 frame/seg (recomendado)</option>
              <option value={2.0}>2 frames/seg (más preciso en movimiento)</option>
              <option value={3.0}>3 frames/seg (máximo detalle)</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            {files.length > 0 && (
              <button className="btn btn-ghost btn-sm" onClick={() => setFiles([])}>
                Limpiar cola
              </button>
            )}
            <button
              id="btn-start-analysis"
              className="btn btn-primary"
              disabled={files.length === 0 || uploading}
              onClick={handleStart}
            >
              <Sparkles size={16} />
              {uploading
                ? 'Subiendo...'
                : `Analizar y filtrar ${files.length > 0 ? `(${files.length})` : ''}`}
            </button>
          </div>
        </div>

        {files.length > 0 && (
          <div className="file-queue-list">
            {files.map((file, idx) => (
              <div key={`${file.name}-${idx}`} className="file-chip">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
                  {isVideoFile(file) ? (
                    <Film size={15} style={{ flexShrink: 0, color: 'var(--accent)' }} />
                  ) : (
                    <ImageIcon size={15} style={{ flexShrink: 0, color: 'var(--success)' }} />
                  )}
                  <div style={{ minWidth: 0 }}>
                    <div className="file-chip-name">{file.name}</div>
                    <div style={{ fontSize: '0.73rem', color: 'var(--ink-muted)' }}>
                      {formatBytes(file.size)}
                    </div>
                  </div>
                </div>
                <button
                  aria-label="Quitar"
                  className="btn btn-ghost btn-sm"
                  style={{ padding: 4 }}
                  onClick={() => removeFile(idx)}
                >
                  <X size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {activeJob && (activeJob.status === 'queued' || activeJob.status === 'running') && (
        <div
          className="panel"
          style={{
            background: 'var(--bg-darkroom)',
            color: 'var(--ink-inverse)',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: 10,
              gap: 12,
              flexWrap: 'wrap',
            }}
          >
            <div style={{ fontWeight: 600 }}>{activeJob.current_step}</div>
            <span className="mono" style={{ fontSize: '0.88rem' }}>
              {Math.round(activeJob.progress * 100)}%
            </span>
          </div>
          <div className="progress-track" style={{ maxWidth: '100%' }}>
            <div
              className="progress-fill"
              style={{ width: `${Math.max(4, Math.round(activeJob.progress * 100))}%` }}
            />
          </div>
        </div>
      )}

      {lastFinishedJob?.summary && (
        <div className="panel">
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: 12,
              marginBottom: 16,
            }}
          >
            <div>
              <h2 className="section-title" style={{ marginBottom: 4 }}>
                Resumen del filtrado inteligente
              </h2>
              <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--ink-secondary)' }}>
                {lastFinishedJob.summary.total_frames_analyzed} frames analizados ·{' '}
                <strong>{lastFinishedJob.summary.kept_detections} mejores capturas</strong>{' '}
                extraídas · {lastFinishedJob.summary.discarded_detections} descartadas por menor
                calidad o redundancia · {lastFinishedJob.summary.new_persons} persona(s) nueva(s)
              </p>
            </div>
            <button id="btn-go-people-after-upload" className="btn btn-dark" onClick={onGoPeople}>
              <Layers size={15} /> Ver personas extraídas
            </button>
          </div>

          <div className="result-list">
            {lastFinishedJob.summary.media_results.map((item) => {
              const isUseful = item.verdict === 'useful';
              const isDiscarded = item.verdict === 'discarded';
              return (
                <div key={item.media_id} className="result-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    {isUseful ? (
                      <CheckCircle2 size={18} style={{ color: 'var(--success)', flexShrink: 0 }} />
                    ) : (
                      <XCircle
                        size={18}
                        style={{
                          color: isDiscarded ? 'var(--warning)' : 'var(--ink-muted)',
                          flexShrink: 0,
                        }}
                      />
                    )}
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.92rem' }}>{item.filename}</div>
                      <div style={{ fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
                        {item.verdict_reason}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span
                      className={`badge ${
                        isUseful
                          ? 'badge-success'
                          : isDiscarded
                            ? 'badge-warning'
                            : ''
                      }`}
                    >
                      {isUseful ? 'Útil' : isDiscarded ? 'Descartada' : 'Sin personas'}
                    </span>
                    <span className="badge mono">
                      {item.kept}/{item.detections}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
