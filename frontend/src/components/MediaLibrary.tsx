import { useState } from 'react';
import { Film, Image as ImageIcon, Play, Trash2, Upload } from 'lucide-react';
import { api, assetUrl, formatBytes, formatTime } from '../api';
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

  const filtered = media.filter((m) => {
    if (filter === 'video') return m.media_type === 'video';
    if (filter === 'image') return m.media_type === 'image';
    if (filter === 'useful') return m.verdict === 'useful';
    if (filter === 'discarded') return m.verdict === 'discarded' || m.verdict === 'no_people';
    return true;
  });

  const handleDeleteSource = async (item: MediaItem, keepAssets: boolean) => {
    const msg = keepAssets
      ? `¿Borrar el archivo original "${item.original_filename}" para liberar espacio?\n\nLas imágenes y recortes PNG de tus amigos ya extraídos SE CONSERVARÁN.`
      : `¿Eliminar "${item.original_filename}" y TODAS las detecciones asociadas?`;
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
          Sube tus fotos y vídeos para indexar a todos tus amigos. Si mantienes los vídeos aquí,
          podrás buscar cualquier foto y saltar al instante al segundo exacto donde aparece.
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
            Reproduce vídeos con marcadores de personas o libera espacio borrando vídeos originales
            sin perder los retratos y recortes PNG guardados.
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

      <div className="media-grid">
        {filtered.map((item) => {
          const deleted = Boolean(item.source_deleted);
          return (
            <article key={item.id} className="media-card">
              <div
                className="media-card-thumb"
                onClick={() => {
                  if (!deleted) onOpenMediaAt(item.id, 0);
                }}
              >
                {item.thumb_path ? (
                  <img src={assetUrl.thumb(item.thumb_path)} alt={item.original_filename} loading="lazy" />
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
              </div>

              <div className="media-card-body">
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
                        onClick={() => onSelectPerson(p.id)}
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
                    borderTop: '1px solid var(--border-hairline)',
                    fontSize: '0.76rem',
                    color: 'var(--ink-muted)',
                  }}
                >
                  <span>
                    {formatBytes(item.file_size)}
                    {deleted && ' · Original borrado'}
                  </span>
                  <div style={{ display: 'flex', gap: 4 }}>
                    {!deleted && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ padding: '4px 8px' }}
                        onClick={() => handleDeleteSource(item, true)}
                        title="Borrar solo el archivo original (mantiene retratos y recortes PNG)"
                      >
                        Liberar espacio
                      </button>
                    )}
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '4px 6px', color: 'var(--danger)' }}
                      onClick={() => handleDeleteSource(item, false)}
                      title="Eliminar completamente de la biblioteca"
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
