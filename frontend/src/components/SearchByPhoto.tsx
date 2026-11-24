import { useCallback, useEffect, useRef, useState } from 'react';
import { Clipboard, Clock, Film, Image as ImageIcon, Play, Search, Upload, UserCheck } from 'lucide-react';
import { api, assetUrl, formatTime } from '../api';
import { extractImagesFromPasteEvent, readImagesFromClipboardApi } from '../clipboard';
import type { SearchResult } from '../types';

interface Props {
  onSelectPerson: (personId: string) => void;
  onOpenMediaAt: (mediaId: string, timestampSec: number) => void;
  onToast: (msg: string) => void;
}

export function SearchByPhoto({ onSelectPerson, onOpenMediaAt, onToast }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [queryFile, setQueryFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [threshold, setThreshold] = useState<number>(0.42);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);

  const runSearch = useCallback(
    async (file: File, faceIndex = 0, simThreshold = threshold) => {
      setLoading(true);
      try {
        const data = await api.searchByImage(file, faceIndex, simThreshold);
        setResult(data);
      } catch (err) {
        onToast(err instanceof Error ? err.message : 'Error al buscar');
      } finally {
        setLoading(false);
      }
    },
    [threshold, onToast]
  );

  const handleFilePick = useCallback(
    (file: File) => {
      setQueryFile(file);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      void runSearch(file, 0, threshold);
    },
    [runSearch, threshold]
  );

  // Global Ctrl+V / Cmd+V paste while on Search tab
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return;
      const pasted = extractImagesFromPasteEvent(e);
      if (pasted.length > 0) {
        e.preventDefault();
        handleFilePick(pasted[0]);
        onToast('Imagen pegada del portapapeles. Buscando...');
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [handleFilePick, onToast]);

  const handlePasteButton = async () => {
    try {
      const pasted = await readImagesFromClipboardApi();
      handleFilePick(pasted[0]);
      onToast('Imagen pegada del portapapeles. Buscando...');
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'No se pudo leer el portapapeles');
    }
  };

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 22 }}>
      <header className="page-header" style={{ marginBottom: 0 }}>
        <div>
          <h1 className="page-title">Buscar persona por foto</h1>
          <p className="page-subtitle">
            Sube o pega del portapapeles (<kbd>Ctrl+V</kbd> / <kbd>⌘V</kbd>) una imagen de
            referencia para saber al instante en qué fotos y segundos exactos de vídeo aparece.
          </p>
        </div>
      </header>

      <div className="panel search-layout">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div
            id="search-dropzone"
            className="dropzone"
            style={{ padding: previewUrl ? 14 : '34px 18px' }}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files?.[0];
              if (f) handleFilePick(f);
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFilePick(f);
                e.target.value = '';
              }}
            />
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Imagen de búsqueda"
                style={{
                  width: '100%',
                  maxHeight: 280,
                  objectFit: 'contain',
                  borderRadius: 10,
                }}
              />
            ) : (
              <>
                <div className="empty-studio-icon" style={{ marginBottom: 10 }}>
                  <Upload size={22} />
                </div>
                <div style={{ fontWeight: 600 }}>Elige, arrastra o pega una foto</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--ink-muted)', marginTop: 4 }}>
                  También funciona con fotos grupales y Ctrl+V / ⌘V
                </div>
              </>
            )}
          </div>

          <button
            id="btn-paste-search"
            type="button"
            className="btn"
            onClick={() => void handlePasteButton()}
          >
            <Clipboard size={15} /> Pegar foto del portapapeles
          </button>

          <div className="field">
            <label htmlFor="range-threshold">
              Sensibilidad de coincidencia: <span className="mono">{Math.round(threshold * 100)}%</span>
            </label>
            <input
              id="range-threshold"
              type="range"
              min={0.3}
              max={0.7}
              step={0.02}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <span className="field-hint">
              Más bajo encuentra más ángulos/iluminaciones; más alto es más estricto.
            </span>
          </div>

          {queryFile && (
            <button
              id="btn-rerun-search"
              className="btn btn-primary"
              disabled={loading}
              onClick={() => runSearch(queryFile, result?.selected_face_index ?? 0, threshold)}
            >
              <Search size={16} />
              {loading ? 'Buscando...' : 'Volver a buscar'}
            </button>
          )}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {!result && !loading && (
            <div style={{ padding: '26px 12px', color: 'var(--ink-secondary)' }}>
              Sube una foto a la izquierda para buscar coincidencias en todos los perfiles y en
              todos los frames analizados de tus fotos y vídeos.
            </div>
          )}

          {loading && (
            <div style={{ padding: '26px 12px', color: 'var(--ink-secondary)' }}>
              Analizando rostro y comparando contra tu biblioteca local...
            </div>
          )}

          {result && !loading && (
            <>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 10,
                }}
              >
                <div style={{ fontWeight: 600 }}>{result.message}</div>
                {result.found_faces > 1 && queryFile && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
                      Rostro en la foto:
                    </span>
                    {result.query_faces.map((qf) => (
                      <button
                        key={qf.index}
                        className={`btn btn-sm ${
                          qf.index === result.selected_face_index ? 'btn-primary' : ''
                        }`}
                        onClick={() => runSearch(queryFile, qf.index, threshold)}
                      >
                        #{qf.index + 1}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {result.matched_persons.length > 0 && (
                <div>
                  <h2 className="section-title">Perfiles coincidentes</h2>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                    {result.matched_persons.map((p) => {
                      const img =
                        assetUrl.portrait(p.cover_detection?.portrait_path) ||
                        assetUrl.thumb(p.cover_detection?.thumb_path);
                      return (
                        <button
                          key={p.id}
                          className="panel"
                          style={{
                            padding: 10,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 12,
                            textAlign: 'left',
                            minWidth: 230,
                          }}
                          onClick={() => onSelectPerson(p.id)}
                        >
                          {img ? (
                            <img
                              src={img}
                              alt={p.name}
                              style={{
                                width: 52,
                                height: 52,
                                borderRadius: 10,
                                objectFit: 'cover',
                              }}
                            />
                          ) : (
                            <UserCheck size={24} />
                          )}
                          <div>
                            <div style={{ fontWeight: 700 }}>{p.name}</div>
                            <div className="badge badge-success mono" style={{ marginTop: 4 }}>
                              {p.similarity}% similitud
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {result.matched_media.length > 0 && (
                <div>
                  <h2 className="section-title">Fotos y vídeos donde aparece</h2>
                  <div className="result-list">
                    {result.matched_media.map((m) => (
                      <div key={m.media_id} className="result-row" style={{ alignItems: 'flex-start' }}>
                        <div style={{ flex: 1, minWidth: 220 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                            {m.media_type === 'video' ? (
                              <Film size={16} style={{ color: 'var(--accent)' }} />
                            ) : (
                              <ImageIcon size={16} style={{ color: 'var(--success)' }} />
                            )}
                            <strong>{m.filename}</strong>
                            <span className="badge badge-success mono">
                              {m.best_similarity}% máx.
                            </span>
                            {m.source_deleted && (
                              <span className="badge badge-warning">Vídeo borrado (solo índice)</span>
                            )}
                          </div>

                          <div className="timestamp-chips" style={{ marginTop: 10 }}>
                            {m.timestamps.slice(0, 14).map((ts) => (
                              <button
                                key={ts.detection_id}
                                className="ts-chip"
                                disabled={m.source_deleted}
                                onClick={() => onOpenMediaAt(m.media_id, ts.timestamp_sec)}
                                title={
                                  m.source_deleted
                                    ? 'El archivo original fue eliminado'
                                    : 'Abrir en este momento'
                                }
                              >
                                {m.media_type === 'video' ? (
                                  <>
                                    <Play size={11} /> {formatTime(ts.timestamp_sec)}
                                  </>
                                ) : (
                                  <>
                                    <Clock size={11} /> Ver foto
                                  </>
                                )}
                                <span style={{ color: 'var(--ink-muted)' }}>
                                  ({ts.similarity}%)
                                </span>
                              </button>
                            ))}
                            {m.timestamps.length > 14 && (
                              <span className="badge">+{m.timestamps.length - 14} más</span>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
