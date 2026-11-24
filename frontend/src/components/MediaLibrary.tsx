import { useEffect, useRef, useState } from 'react';
import { Check, Clock, Film, Image as ImageIcon, Pin, Play, Trash2, Upload } from 'lucide-react';
import { api, assetUrl, formatBytes, formatTime, formatTimeRemaining } from '../api';
import type { MediaItem } from '../types';

interface Props {
  media: MediaItem[];
  onOpenMediaAt: (mediaId: string, timestampSec: number) => void;
  onSelectPerson: (personId: string) => void;
  onGoUpload: () => void;
  onRefresh: () => void;
  onToast: (msg: string) => void;
}

export function MediaLibrary({
  media,
  onOpenMediaAt,
  onSelectPerson,
  onGoUpload,
  onRefresh,
  onToast,
}: Props) {
  const [filter, setFilter] = useState<'all' | 'video' | 'image' | 'useful' | 'discarded'>('all');
  const [busyId, setBusyId] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const lastSelectedIdRef = useRef<string | null>(null);

  const filtered = media.filter((m) => {
    if (filter === 'video') return m.media_type === 'video';
    if (filter === 'image') return m.media_type === 'image';
    if (filter === 'useful') return m.verdict === 'useful';
    if (filter === 'discarded') return m.verdict === 'discarded' || m.verdict === 'no_people';
    return true;
  });

  const toggleSelect = (
    id: string,
    e?: { shiftKey?: boolean; preventDefault?: () => void }
  ) => {
    if (e?.shiftKey) {
      e.preventDefault?.();
      window.getSelection()?.removeAllRanges();
      const currentIdx = filtered.findIndex((m) => m.id === id);
      let anchorId = lastSelectedIdRef.current;
      if (!anchorId || !filtered.some((m) => m.id === anchorId)) {
        const lastExisting = [...selectedIds]
          .reverse()
          .find((sid) => filtered.some((m) => m.id === sid));
        anchorId = lastExisting || null;
      }
      const anchorIdx = anchorId ? filtered.findIndex((m) => m.id === anchorId) : -1;

      if (currentIdx !== -1 && anchorIdx !== -1) {
        const start = Math.min(anchorIdx, currentIdx);
        const end = Math.max(anchorIdx, currentIdx);
        const rangeIds = filtered.slice(start, end + 1).map((m) => m.id);
        setSelectedIds((prev) => {
          const next = new Set(prev);
          for (const rid of rangeIds) next.add(rid);
          return Array.from(next);
        });
        lastSelectedIdRef.current = id;
        return;
      }
    }

    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
    lastSelectedIdRef.current = id;
  };

  const handleBulkDelete = async (keepAssets: boolean) => {
    if (selectedIds.length === 0 || bulkBusy) return;
    const count = selectedIds.length;
    const msg = keepAssets
      ? `¿Borrar los archivos originales de ${count} elemento(s) seleccionado(s)?\n\nLos retratos y recortes PNG ya extraídos SE CONSERVARÁN.`
      : `¿Eliminar ${count} archivo(s) seleccionado(s) y sus fotos extraídas?\n\nLos perfiles de las personas y su reconocimiento facial SE CONSERVARÁN.`;
    if (!window.confirm(msg)) return;
    setBulkBusy(true);
    try {
      for (const id of selectedIds) {
        await api.deleteMedia(id, keepAssets);
      }
      setSelectedIds([]);
      lastSelectedIdRef.current = null;
      onToast(
        keepAssets
          ? `${count} original(es) limpiado(s) (recortes conservados)`
          : `${count} archivo(s) eliminado(s)`
      );
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error eliminando archivos');
    } finally {
      setBulkBusy(false);
    }
  };

  useEffect(() => {
    if (selectedIds.length === 0) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (bulkBusy) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)
      ) {
        return;
      }
      if (e.key === 'Escape') {
        setSelectedIds([]);
        lastSelectedIdRef.current = null;
      } else if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault();
        void handleBulkDelete(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  const handleToggleKeep = async (item: MediaItem) => {
    setBusyId(item.id);
    try {
      const nextKeep = !item.keep_forever;
      await api.setMediaRetention(item.id, nextKeep);
      onToast(
        nextKeep
          ? `"${item.original_filename}" marcado para mantenerse sin caducidad.`
          : `"${item.original_filename}" se auto-borrará en 1 hora (sus recortes se conservan).`
      );
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error cambiando retención');
    } finally {
      setBusyId(null);
    }
  };

  const handleDeleteSource = async (item: MediaItem, keepAssets: boolean) => {
    const msg = keepAssets
      ? `¿Borrar el archivo original "${item.original_filename}" ahora?\n\nLas imágenes y recortes PNG de tus amigos ya extraídos SE CONSERVARÁN.`
      : `¿Eliminar "${item.original_filename}" y sus fotos extraídas?\n\nLos perfiles de las personas y su reconocimiento facial SE CONSERVARÁN para futuras fotos y vídeos.`;
    if (!window.confirm(msg)) return;
    try {
      const res = await api.deleteMedia(item.id, keepAssets);
      onToast(res.message);
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error eliminando archivo');
    }
  };

  if (media.length === 0) {
    return (
      <section className="empty-studio">
        <div className="empty-studio-icon">
          <Film size={25} />
        </div>
        <h2>Biblioteca vacía</h2>
        <p>
          Sube tus fotos y vídeos para analizarlos. Por defecto los originales se limpian solos tras
          1 hora (conservando siempre lo generado), o puedes pulsar «Mantener» en cualquier archivo.
        </p>
        <button className="btn btn-primary" onClick={onGoUpload}>
          <Upload size={16} /> Subir archivos
        </button>
      </section>
    );
  }

  return (
    <section>
      <header className="page-header">
        <div>
          <h1 className="page-title">Biblioteca de fotos y vídeos</h1>
          <p className="page-subtitle">
            Los originales expiran por defecto en 1 hora para ahorrar espacio (los retratos y
            recortes PNG nunca se borran). Pulsa <strong>Mantener</strong> en los vídeos que
            quieras conservar o usa <strong>Shift + clic</strong> para seleccionar varios y
            eliminarlos a la vez.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {(
            [
              ['all', 'Todos'],
              ['video', 'Vídeos'],
              ['image', 'Fotos'],
              ['useful', 'Útiles'],
              ['discarded', 'Descartados'],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              className={`btn btn-sm ${filter === key ? 'btn-dark' : ''}`}
              onClick={() => setFilter(key)}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      {selectedIds.length > 0 && (
        <div
          className="panel"
          style={{
            marginBottom: 18,
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 10,
            background: 'var(--accent-soft)',
            borderColor: 'hsla(16, 66%, 47%, 0.25)',
          }}
        >
          <span style={{ fontWeight: 600, color: 'var(--accent-ink)' }}>
            {selectedIds.length} archivo(s) seleccionado(s)
            <span style={{ fontWeight: 400, fontSize: '0.78rem', marginLeft: 8, opacity: 0.85 }}>
              (Shift + clic para rango · Supr para eliminar)
            </span>
          </span>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {selectedIds.length < filtered.length && (
              <button
                className="btn btn-sm"
                disabled={bulkBusy}
                onClick={() => {
                  setSelectedIds(filtered.map((m) => m.id));
                  if (filtered.length > 0) {
                    lastSelectedIdRef.current = filtered[filtered.length - 1].id;
                  }
                }}
              >
                Seleccionar todos ({filtered.length})
              </button>
            )}
            <button
              className="btn btn-sm"
              disabled={bulkBusy}
              onClick={() => void handleBulkDelete(true)}
              title="Borra solo los archivos originales, conservando retratos y recortes PNG"
            >
              Limpiar originales
            </button>
            <button
              className="btn btn-danger btn-sm"
              disabled={bulkBusy}
              onClick={() => void handleBulkDelete(false)}
            >
              <Trash2 size={14} /> Eliminar ({selectedIds.length})
            </button>
            <button
              className="btn btn-sm"
              onClick={() => {
                setSelectedIds([]);
                lastSelectedIdRef.current = null;
              }}
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      <div className="media-grid">
        {filtered.map((item) => {
          const deleted = Boolean(item.source_deleted);
          const keepForever = Boolean(item.keep_forever);
          const remaining = !deleted && !keepForever ? formatTimeRemaining(item.expires_at) : null;
          const isSelected = selectedIds.includes(item.id);

          return (
            <article
              key={item.id}
              className={`media-card ${isSelected ? 'selected' : ''}`}
              onMouseDown={(e) => {
                if (e.shiftKey) e.preventDefault();
              }}
            >
              <div
                className="media-card-thumb"
                onClick={(e) => {
                  if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    toggleSelect(item.id, e);
                    return;
                  }
                  if (!deleted) onOpenMediaAt(item.id, 0);
                }}
              >
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
                    {item.media_type === 'video' ? <Film size={30} /> : <ImageIcon size={30} />}
                  </div>
                )}

                <div className="card-overlay-top">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <button
                      aria-label={`Seleccionar ${item.original_filename}`}
                      title="Clic para seleccionar · Shift + clic para seleccionar un rango"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleSelect(item.id, e);
                      }}
                      style={{
                        width: 24,
                        height: 24,
                        borderRadius: 6,
                        border: isSelected ? 'none' : '1.5px solid rgba(255,255,255,0.85)',
                        background: isSelected ? 'var(--accent)' : 'rgba(20,18,16,0.45)',
                        color: '#fff',
                        display: 'grid',
                        placeItems: 'center',
                      }}
                    >
                      {isSelected && <Check size={14} />}
                    </button>
                    <span className="badge badge-dark">
                      {item.media_type === 'video' ? (
                        <>
                          <Play size={10} />{' '}
                          {item.duration_sec ? formatTime(item.duration_sec) : 'Vídeo'}
                        </>
                      ) : (
                        'Foto'
                      )}
                    </span>
                  </div>
                  <span
                    className={`badge ${
                      item.verdict === 'useful'
                        ? 'badge-success'
                        : item.verdict === 'discarded'
                          ? 'badge-warning'
                          : 'badge-dark'
                    }`}
                  >
                    {item.verdict === 'useful'
                      ? 'Útil'
                      : item.verdict === 'discarded'
                        ? 'Descartada'
                        : item.verdict === 'no_people'
                          ? 'Sin rostros'
                          : 'Procesando'}
                  </span>
                </div>

                <div className="card-overlay-bottom">
                  {deleted ? (
                    <span className="badge badge-dark">Original borrado · Recortes OK</span>
                  ) : keepForever ? (
                    <span className="badge badge-success">
                      <Pin size={11} /> Guardado
                    </span>
                  ) : (
                    <span className="badge badge-warning">
                      <Clock size={11} /> {remaining ? `Expira en ${remaining}` : 'Temporal 1h'}
                    </span>
                  )}
                </div>
              </div>

              <div
                className="media-card-body"
                onClick={(e) => {
                  if (e.shiftKey || e.ctrlKey || e.metaKey) {
                    e.preventDefault();
                    toggleSelect(item.id, e);
                  }
                }}
              >
                <div
                  style={{
                    fontWeight: 700,
                    fontSize: '0.92rem',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={item.original_filename}
                >
                  {item.original_filename}
                </div>

                <div style={{ fontSize: '0.79rem', color: 'var(--ink-secondary)' }}>
                  {item.verdict_reason || `${item.detections_count} detecciones`}
                </div>

                {item.persons && item.persons.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                    {item.persons.map((p) => (
                      <button
                        key={p.id}
                        className="ts-chip"
                        style={{ fontFamily: 'var(--font-sans)' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectPerson(p.id);
                        }}
                      >
                        {p.name}
                      </button>
                    ))}
                  </div>
                )}

                <div
                  style={{
                    marginTop: 'auto',
                    paddingTop: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 6,
                    borderTop: '1px solid var(--border-hairline)',
                    fontSize: '0.76rem',
                    color: 'var(--ink-muted)',
                  }}
                >
                  <span>{formatBytes(item.file_size)}</span>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    {!deleted && (
                      <button
                        className={`btn btn-sm ${keepForever ? 'btn-ghost' : 'btn-primary'}`}
                        style={{ padding: '4px 9px', fontSize: '0.76rem' }}
                        disabled={busyId === item.id}
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleToggleKeep(item);
                        }}
                      >
                        {keepForever ? (
                          <>
                            <Clock size={12} /> 1h
                          </>
                        ) : (
                          <>
                            <Pin size={12} /> Mantener
                          </>
                        )}
                      </button>
                    )}
                    {!deleted && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '4px 7px', fontSize: '0.75rem' }}
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleDeleteSource(item, true);
                        }}
                        title="Borrar original ahora (mantiene retratos y recortes PNG)"
                      >
                        Limpiar
                      </button>
                    )}
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '4px 6px', color: 'var(--danger)' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleDeleteSource(item, false);
                      }}
                      title="Eliminar registro y detecciones"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
