import { useState } from 'react';
import {
  Check,
  Download,
  Edit3,
  GitMerge,
  Image as ImageIcon,
  Scissors,
  Search,
  Trash2,
  Upload,
  Users,
  X,
} from 'lucide-react';
import { api, assetUrl } from '../api';
import type { PersonSummary } from '../types';

interface Props {
  persons: PersonSummary[];
  loading: boolean;
  onSelectPerson: (personId: string) => void;
  onGoUpload: () => void;
  onRefresh: () => void;
  onToast: (msg: string) => void;
}

export function PeopleGallery({
  persons,
  loading,
  onSelectPerson,
  onGoUpload,
  onRefresh,
  onToast,
}: Props) {
  const [query, setQuery] = useState('');
  const [previewMode, setPreviewMode] = useState<'portrait' | 'cutout'>('portrait');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState('');
  const [busy, setBusy] = useState(false);

  const handleQuickRename = async (personId: string) => {
    const trimmed = nameDraft.trim();
    if (!trimmed) {
      setEditingId(null);
      return;
    }
    try {
      await api.updatePerson(personId, { name: trimmed });
      setEditingId(null);
      onToast(`Guardado como "${trimmed}". Se reconocerá automáticamente en futuras fotos y vídeos.`);
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al guardar nombre');
    }
  };

  const filtered = persons.filter((p) =>
    p.name.toLowerCase().includes(query.trim().toLowerCase())
  );

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleMerge = async () => {
    if (selectedIds.length < 2) return;
    const [targetId, ...sources] = selectedIds;
    const targetName = persons.find((p) => p.id === targetId)?.name || 'esta persona';
    if (
      !window.confirm(
        `¿Fusionar ${selectedIds.length} perfiles en "${targetName}"? Todas sus capturas y apariciones se combinarán.`
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      await api.mergePersons(targetId, sources);
      setSelectedIds([]);
      onToast(`Perfiles fusionados en "${targetName}"`);
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al fusionar');
    } finally {
      setBusy(false);
    }
  };

  const handleExport = async () => {
    setBusy(true);
    try {
      await api.exportZip({
        person_ids: selectedIds.length > 0 ? selectedIds : null,
        include_portraits: true,
        include_cutouts: true,
        include_crops: true,
        include_originals: false,
        only_kept: true,
      });
      onToast('Pack ZIP listo para tu videojuego descargado.');
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al exportar');
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (
      !window.confirm(
        `¿Eliminar ${selectedIds.length} perfil(es) seleccionado(s)? (Útil para quitar desconocidos del fondo)`
      )
    ) {
      return;
    }
    setBusy(true);
    try {
      for (const id of selectedIds) {
        await api.deletePerson(id);
      }
      setSelectedIds([]);
      onToast('Perfiles eliminados');
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al eliminar');
    } finally {
      setBusy(false);
    }
  };

  if (!loading && persons.length === 0) {
    return (
      <section className="empty-studio">
        <div className="empty-studio-icon">
          <Users size={26} />
        </div>
        <h2>Aún no hay amigos detectados</h2>
        <p>
          Sube tus fotos o vídeos para analizarlos frame a frame en local. FriendFinder agrupará a
          cada persona automáticamente, extraerá sus mejores retratos y recortes PNG sin otras
          personas, y descartará las tomas de menor calidad.
        </p>
        <button id="btn-empty-upload" className="btn btn-primary" onClick={onGoUpload}>
          <Upload size={17} />
          Subir fotos o vídeos
        </button>
      </section>
    );
  }

  return (
    <section>
      <header className="page-header">
        <div>
          <h1 className="page-title">Personas detectadas</h1>
          <p className="page-subtitle">
            Cada ficha reúne automáticamente el mejor retrato, el mejor recorte de cuerpo entero sin
            otras personas (PNG) y todas sus apariciones en tus vídeos y fotos.
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <Search
              size={15}
              style={{
                position: 'absolute',
                left: 11,
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--ink-muted)',
              }}
            />
            <input
              id="input-filter-people"
              className="input"
              style={{ paddingLeft: 33, width: 200 }}
              placeholder="Filtrar por nombre..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>

          <div
            style={{
              display: 'inline-flex',
              background: 'var(--bg-subtle)',
              padding: 3,
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-hairline)',
            }}
          >
            <button
              id="btn-mode-portrait"
              className={`btn btn-sm ${previewMode === 'portrait' ? '' : 'btn-ghost'}`}
              style={{ border: 'none' }}
              onClick={() => setPreviewMode('portrait')}
            >
              <ImageIcon size={14} /> Retratos
            </button>
            <button
              id="btn-mode-cutout"
              className={`btn btn-sm ${previewMode === 'cutout' ? '' : 'btn-ghost'}`}
              style={{ border: 'none' }}
              onClick={() => setPreviewMode('cutout')}
            >
              <Scissors size={14} /> Recortes PNG
            </button>
          </div>

          <button
            id="btn-export-all"
            className="btn btn-dark"
            disabled={busy || persons.length === 0}
            onClick={handleExport}
          >
            <Download size={16} />
            {selectedIds.length > 0
              ? `Exportar (${selectedIds.length}) ZIP`
              : 'Exportar Pack ZIP'}
          </button>
        </div>
      </header>

      <div className="studio-strip">
        <span className="studio-strip-item">
          <Users size={15} style={{ color: 'var(--accent)' }} />
          <strong>{persons.length}</strong> persona(s)
        </span>
        <span className="studio-strip-item">
          <ImageIcon size={15} style={{ color: 'var(--success)' }} />
          <strong>{persons.reduce((a, p) => a + p.kept_count, 0)}</strong> mejores capturas
        </span>
        <span className="studio-strip-item">
          <Scissors size={15} style={{ color: 'var(--ink-primary)' }} />
          <strong>{persons.reduce((a, p) => a + p.cutout_count, 0)}</strong> recortes PNG sin fondo
        </span>
      </div>

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
            {selectedIds.length} persona(s) seleccionada(s)
          </span>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {selectedIds.length >= 2 && (
              <button
                id="btn-merge-selected"
                className="btn btn-primary btn-sm"
                disabled={busy}
                onClick={handleMerge}
              >
                <GitMerge size={14} /> Fusionar en una misma persona
              </button>
            )}
            <button
              id="btn-delete-selected"
              className="btn btn-danger btn-sm"
              disabled={busy}
              onClick={handleDeleteSelected}
            >
              <Trash2 size={14} /> Eliminar seleccionados
            </button>
            <button
              className="btn btn-sm"
              onClick={() => setSelectedIds([])}
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      <div className="people-grid">
        {filtered.map((person) => {
          const isSelected = selectedIds.includes(person.id);
          const showCutout = previewMode === 'cutout' && Boolean(person.best_body?.cutout_path);
          const imgSrc = showCutout
            ? assetUrl.cutout(person.best_body?.cutout_path)
            : assetUrl.portrait(person.cover_detection?.portrait_path) ||
              assetUrl.thumb(person.cover_detection?.thumb_path) ||
              assetUrl.portrait(person.best_portrait?.portrait_path) ||
              assetUrl.thumb(person.best_portrait?.thumb_path);

          return (
            <article
              key={person.id}
              className={`person-card ${isSelected ? 'selected' : ''}`}
            >
              <div
                className={`person-card-media ${showCutout ? 'checkerboard-bg cutout-preview' : ''}`}
                onClick={() => onSelectPerson(person.id)}
              >
                {imgSrc ? (
                  <img src={imgSrc} alt={person.name} loading="lazy" />
                ) : (
                  <div
                    style={{
                      width: '100%',
                      height: '100%',
                      display: 'grid',
                      placeItems: 'center',
                      color: 'var(--ink-muted)',
                    }}
                  >
                    <Users size={34} />
                  </div>
                )}

                <div className="card-overlay-top">
                  <button
                    aria-label={`Seleccionar ${person.name}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSelect(person.id);
                    }}
                    style={{
                      width: 26,
                      height: 26,
                      borderRadius: 7,
                      border: isSelected ? 'none' : '1.5px solid rgba(255,255,255,0.85)',
                      background: isSelected ? 'var(--accent)' : 'rgba(20,18,16,0.42)',
                      color: '#fff',
                      display: 'grid',
                      placeItems: 'center',
                    }}
                  >
                    {isSelected && <Check size={15} />}
                  </button>

                  <span className="badge badge-dark mono">
                    ★ {person.max_portrait_score.toFixed(0)}
                  </span>
                </div>

                <div className="card-overlay-bottom">
                  {person.cutout_count > 0 && (
                    <span className="badge badge-dark">
                      <Scissors size={11} /> {person.cutout_count} PNG
                    </span>
                  )}
                  {person.max_body_score > 0 && (
                    <span className="badge badge-dark mono">
                      Cuerpo {person.max_body_score.toFixed(0)}
                    </span>
                  )}
                </div>
              </div>

              <div
                className="person-card-body"
                style={{ cursor: editingId === person.id ? 'default' : 'pointer' }}
                onClick={() => {
                  if (editingId !== person.id) onSelectPerson(person.id);
                }}
              >
                {editingId === person.id ? (
                  <div
                    style={{ display: 'flex', gap: 5, alignItems: 'center' }}
                    onClick={(e) => e.stopPropagation()}
                  >
                    <input
                      className="input"
                      style={{ padding: '5px 8px', fontSize: '0.86rem' }}
                      value={nameDraft}
                      placeholder="Nombre del amigo..."
                      autoFocus
                      onChange={(e) => setNameDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') void handleQuickRename(person.id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                    />
                    <button
                      aria-label="Guardar nombre"
                      className="btn btn-primary btn-sm"
                      style={{ padding: '6px 8px' }}
                      onClick={() => handleQuickRename(person.id)}
                    >
                      <Check size={14} />
                    </button>
                    <button
                      aria-label="Cancelar"
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '6px 6px' }}
                      onClick={() => setEditingId(null)}
                    >
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <div className="person-card-name-row">
                    <h3 className="person-card-name">{person.name}</h3>
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '3px 7px', fontSize: '0.75rem', flexShrink: 0 }}
                      title="Guardar con nombre para reconocimiento automático"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingId(person.id);
                        setNameDraft(/^Persona \d+$/.test(person.name) ? '' : person.name);
                      }}
                    >
                      <Edit3 size={13} />
                      {/^Persona \d+$/.test(person.name) ? 'Nombrar' : ''}
                    </button>
                  </div>
                )}
                <div className="person-meta-row" style={{ alignItems: 'center' }}>
                  <span className="badge badge-success mono">{person.kept_count} mejores</span>
                  <span>{person.media_count} archivo(s)</span>
                  {person.discarded_count > 0 && (
                    <span style={{ color: 'var(--ink-muted)' }}>
                      · {person.discarded_count} desc.
                    </span>
                  )}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
