import { useEffect, useRef, useState } from 'react';
import {
  Check,
  CheckCircle2,
  Clipboard,
  Clock,
  Film,
  Image as ImageIcon,
  Layers,
  Pin,
  Sparkles,
  Trash2,
  Upload,
  UserCheck,
  X,
  XCircle,
} from 'lucide-react';
import { api, assetUrl, formatBytes } from '../api';
import { extractImagesFromPasteEvent, readImagesFromClipboardApi } from '../clipboard';
import type { JobInfo, PersonSummary } from '../types';

interface Props {
  activeJob: JobInfo | null;
  lastFinishedJob: JobInfo | null;
  persons: PersonSummary[];
  defaultSampleFps: number;
  retentionMinutes: number;
  onJobStarted: (jobId: string) => void;
  onDismissSummary: () => void;
  onGoPeople: () => void;
  onPersonsUpdated: () => void;
  onToast: (msg: string) => void;
}

export function UploadStudio({
  activeJob,
  lastFinishedJob,
  persons,
  defaultSampleFps,
  retentionMinutes,
  onJobStarted,
  onDismissSummary,
  onGoPeople,
  onPersonsUpdated,
  onToast,
}: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [sampleFps, setSampleFps] = useState<number>(defaultSampleFps || 1.0);
  // Por defecto false: el archivo subido expira en 1 hora (lo generado se conserva siempre)
  const [keepForever, setKeepForever] = useState<boolean>(false);
  const [uploading, setUploading] = useState(false);
  const [nameDrafts, setNameDrafts] = useState<Record<string, string>>({});
  const [savedNames, setSavedNames] = useState<Record<string, string>>({});
  const [deletingIds, setDeletingIds] = useState<Set<string>>(new Set());

  const livePersonMap = new Map(persons.map((p) => [p.id, p]));

  const handleSavePersonName = async (personId: string, fallbackName: string) => {
    const nextName = (nameDrafts[personId] ?? fallbackName).trim();
    if (!nextName) return;
    try {
      await api.updatePerson(personId, { name: nextName });
      setSavedNames((prev) => ({ ...prev, [personId]: nextName }));
      onToast(`"${nextName}" guardado. Se le asignará este nombre automáticamente en futuras fotos y vídeos.`);
      onPersonsUpdated();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error guardando nombre');
    }
  };

  const handleDeleteSummaryPerson = async (personId: string, name: string) => {
    setDeletingIds((prev) => new Set(prev).add(personId));
    try {
      await api.deletePerson(personId);
      onToast(`"${name}" eliminado.`);
      onPersonsUpdated();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al eliminar persona');
    } finally {
      setDeletingIds((prev) => {
        const next = new Set(prev);
        next.delete(personId);
        return next;
      });
    }
  };

  const retentionLabel =
    retentionMinutes === 60
      ? '1 hora'
      : retentionMinutes < 60
        ? `${retentionMinutes} min`
        : `${Math.round((retentionMinutes / 60) * 10) / 10} h`;

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => `${f.name}-${f.size}`));
      const unique = arr.filter((f) => !existing.has(`${f.name}-${f.size}`));
      return [...prev, ...unique];
    });
  };

  // Global Ctrl+V / Cmd+V clipboard image paste while on Upload tab
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      const pasted = extractImagesFromPasteEvent(e);
      if (pasted.length > 0) {
        e.preventDefault();
        addFiles(pasted);
        onToast(`Pegada(s) ${pasted.length} imagen(es) del portapapeles.`);
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [onToast]);

  const handlePasteButton = async () => {
    try {
      const pasted = await readImagesFromClipboardApi();
      addFiles(pasted);
      onToast(`Pegada(s) ${pasted.length} imagen(es) del portapapeles.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo leer el portapapeles');
    }
  };

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleStart = async () => {
    if (files.length === 0) return;
    setUploading(true);
    try {
      const res = await api.uploadMedia(files, sampleFps, keepForever);
      setFiles([]);
      onJobStarted(res.job_id);
      if (res.skipped.length > 0) {
        onToast(
          `Subidos ${res.uploaded_count} archivo(s). ${res.skipped.length} omitido(s) por formato.`
        );
      } else {
        onToast(
          keepForever
            ? `Analizando ${res.uploaded_count} archivo(s) (originales conservados)...`
            : `Analizando ${res.uploaded_count} archivo(s) (originales se borrarán en ${retentionLabel}; recortes conservados)...`
        );
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
            Sube fotos o vídeos desde tu PC o móvil, o pega una imagen con <kbd>Ctrl+V</kbd> /{' '}
            <kbd>⌘V</kbd>. Extraemos los mejores retratos y recortes PNG sin otras personas.
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
          <div className="dropzone-title">Toca, arrastra o pega tus fotos y vídeos</div>
          <p style={{ margin: '0 0 14px', color: 'var(--ink-secondary)', fontSize: '0.89rem' }}>
            JPG, PNG, WebP, MP4, MOV, MKV, WebM · También puedes pegar con Ctrl+V / ⌘V
          </p>
          <button
            id="btn-paste-upload"
            type="button"
            className="btn btn-sm"
            onClick={(e) => {
              e.stopPropagation();
              void handlePasteButton();
            }}
          >
            <Clipboard size={14} /> Pegar del portapapeles
          </button>
        </div>

        {/* 1-hour auto-expiration policy toggle */}
        <div className="retention-banner">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <div style={{ fontWeight: 700, fontSize: '0.88rem' }}>
              Duración de los archivos subidos
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--ink-secondary)' }}>
              {keepForever
                ? 'Se conservarán los archivos originales para reproducir vídeos y buscar timestamps.'
                : `Los originales se borran solos tras ${retentionLabel}. Los retratos, recortes PNG y perfiles generados NO se borran.`}
            </div>
          </div>

          <div className="retention-segmented" role="group" aria-label="Retención de archivos">
            <button
              id="btn-retention-1h"
              type="button"
              className={`retention-opt ephemeral ${!keepForever ? 'active' : ''}`}
              onClick={() => setKeepForever(false)}
            >
              <Clock size={14} /> Borrar en {retentionLabel}
            </button>
            <button
              id="btn-retention-keep"
              type="button"
              className={`retention-opt keep ${keepForever ? 'active' : ''}`}
              onClick={() => setKeepForever(true)}
            >
              <Pin size={14} /> Mantener originales
            </button>
          </div>
        </div>

        <div
          style={{
            marginTop: 16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 14,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <label htmlFor="select-fps" style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              Velocidad en vídeos:
            </label>
            <select
              id="select-fps"
              className="select"
              style={{ width: 'auto' }}
              value={sampleFps}
              onChange={(e) => setSampleFps(Number(e.target.value))}
            >
              <option value={0.5}>0.5 fps (muy rápido)</option>
              <option value={1.0}>1 fps (recomendado)</option>
              <option value={2.0}>2 fps (más preciso)</option>
              <option value={3.0}>3 fps (máximo detalle)</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            {files.length > 0 && (
              <button className="btn btn-ghost btn-sm" onClick={() => setFiles([])}>
                Limpiar
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

      {lastFinishedJob?.summary && (() => {
        const liveDetected = (lastFinishedJob.summary.detected_persons || []).filter(
          (dp) => livePersonMap.has(dp.id) && !deletingIds.has(dp.id)
        );
        return (
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
                  extraídas · {lastFinishedJob.summary.discarded_detections} descartadas ·{' '}
                  {liveDetected.length} persona(s) activa(s)
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                {liveDetected.length > 0 && (
                  <button id="btn-go-people-after-upload" className="btn btn-dark" onClick={onGoPeople}>
                    <Layers size={15} /> Ver personas extraídas
                  </button>
                )}
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={onDismissSummary}
                  title="Cerrar resumen"
                  aria-label="Cerrar resumen"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            {liveDetected.length > 0 && (
              <div style={{ marginBottom: 18 }}>
                <div
                  style={{
                    fontSize: '0.84rem',
                    fontWeight: 700,
                    color: 'var(--ink-secondary)',
                    marginBottom: 10,
                  }}
                >
                  Personas detectadas (ponles nombre y se reconocerán solas en futuras fotos/vídeos):
                </div>
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                    gap: 10,
                  }}
                >
                  {liveDetected.map((dp) => {
                    const livePerson = livePersonMap.get(dp.id);
                    const currentName = savedNames[dp.id] || livePerson?.name || dp.name;
                    const isDefault = /^Persona \d+$/.test(currentName);
                    const draftVal =
                      nameDrafts[dp.id] !== undefined
                        ? nameDrafts[dp.id]
                        : isDefault
                          ? ''
                          : currentName;
                    const thumbSrc =
                      assetUrl.portrait(livePerson?.cover_detection?.portrait_path) ||
                      assetUrl.thumb(livePerson?.cover_detection?.thumb_path) ||
                      assetUrl.portrait(dp.thumb_path) ||
                      assetUrl.thumb(dp.thumb_path);

                    return (
                      <div
                        key={dp.id}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 10,
                          padding: 10,
                          borderRadius: 'var(--radius-md)',
                          background: 'var(--bg-canvas)',
                          border: '1px solid var(--border-hairline)',
                        }}
                      >
                        {thumbSrc ? (
                          <img
                            src={thumbSrc}
                            alt={currentName}
                            style={{
                              width: 48,
                              height: 48,
                              borderRadius: 10,
                              objectFit: 'cover',
                              flexShrink: 0,
                            }}
                          />
                        ) : (
                          <div
                            style={{
                              width: 48,
                              height: 48,
                              borderRadius: 10,
                              background: 'var(--bg-subtle)',
                              display: 'grid',
                              placeItems: 'center',
                              flexShrink: 0,
                            }}
                          >
                            <UserCheck size={20} />
                          </div>
                        )}
                        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 5 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                            <span
                              className={`badge ${dp.is_new ? 'badge-accent' : 'badge-success'}`}
                              style={{ fontSize: '0.7rem' }}
                            >
                              {dp.is_new ? 'Nueva persona' : `✓ Reconocido: ${currentName}`}
                            </span>
                          </div>
                          <div style={{ display: 'flex', gap: 5 }}>
                            <input
                              className="input"
                              style={{ padding: '5px 8px', fontSize: '0.83rem' }}
                              placeholder="Ej. Carlos, Lucía..."
                              value={draftVal}
                              onChange={(e) =>
                                setNameDrafts((prev) => ({ ...prev, [dp.id]: e.target.value }))
                              }
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') {
                                  void handleSavePersonName(dp.id, currentName);
                                }
                              }}
                            />
                            <button
                              className="btn btn-primary btn-sm"
                              style={{ padding: '5px 9px' }}
                              onClick={() => handleSavePersonName(dp.id, currentName)}
                              title="Guardar nombre"
                            >
                              <Check size={14} />
                            </button>
                            <button
                              className="btn btn-ghost btn-sm"
                              style={{ padding: '5px 7px', color: 'var(--danger)' }}
                              onClick={() => void handleDeleteSummaryPerson(dp.id, currentName)}
                              title="Eliminar esta persona"
                              aria-label={`Eliminar a ${currentName}`}
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

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
        );
      })()}
    </section>
  );
}
