import { useRef, useState } from 'react';
import {
  ArrowRight,
  Check,
  Download,
  Edit3,
  GitMerge,
  GripVertical,
  Image as ImageIcon,
  RefreshCw,
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

function isDefaultPersonName(name: string): boolean {
  return /^Persona \d+$/.test(name.trim());
}

function getPersonInitials(name: string): string {
  return (
    name
      .split(/\s+/)
      .map((w) => w[0] || '')
      .join('')
      .slice(0, 2)
      .toUpperCase() || '?'
  );
}

function getPersonAvatarSrc(person: PersonSummary, mode: 'portrait' | 'cutout' = 'portrait'): string {
  const showCutout = mode === 'cutout' && Boolean(person.best_body?.cutout_path);
  return showCutout
    ? assetUrl.cutout(person.best_body?.cutout_path)
    : assetUrl.portrait(person.cover_detection?.portrait_path) ||
        assetUrl.thumb(person.cover_detection?.thumb_path) ||
        assetUrl.portrait(person.best_portrait?.portrait_path) ||
        assetUrl.thumb(person.best_portrait?.thumb_path);
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

  // Drag & drop merge state (supports both desktop HTML5 DnD and mobile touch pointer drag)
  const [dragSourceId, setDragSourceId] = useState<string | null>(null);
  const [dropTargetId, setDropTargetId] = useState<string | null>(null);
  const [pointerGhost, setPointerGhost] = useState<{ x: number; y: number } | null>(null);
  const [pendingMerge, setPendingMerge] = useState<{
    source: PersonSummary;
    target: PersonSummary;
  } | null>(null);
  const [mergeNameDraft, setMergeNameDraft] = useState('');

  const pointerDragRef = useRef<{
    sourceId: string;
    pointerId: number;
    startX: number;
    startY: number;
    active: boolean;
  } | null>(null);
  const justDraggedUntilRef = useRef(0);

  const openMergeConfirm = (sourceId: string, targetId: string) => {
    if (!sourceId || !targetId || sourceId === targetId) return;
    const source = persons.find((p) => p.id === sourceId);
    const target = persons.find((p) => p.id === targetId);
    if (!source || !target) return;

    // Smartly pre-fill with whichever already has a real custom name (prefer target, else source)
    const preferredName = !isDefaultPersonName(target.name)
      ? target.name
      : !isDefaultPersonName(source.name)
        ? source.name
        : target.name;

    setMergeNameDraft(preferredName);
    setPendingMerge({ source, target });
  };

  const handleConfirmDragMerge = async () => {
    if (!pendingMerge || busy) return;
    const { source, target } = pendingMerge;
    const finalName = mergeNameDraft.trim() || target.name;
    setBusy(true);
    try {
      await api.mergePersons(target.id, [source.id]);
      if (finalName && finalName !== target.name) {
        await api.updatePerson(target.id, { name: finalName });
      }
      setSelectedIds((prev) => prev.filter((id) => id !== source.id));
      setPendingMerge(null);
      onToast(`"${source.name}" y "${target.name}" unidas como "${finalName}".`);
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al unir personas');
    } finally {
      setBusy(false);
    }
  };

  const resetDragState = () => {
    pointerDragRef.current = null;
    setDragSourceId(null);
    setDropTargetId(null);
    setPointerGhost(null);
  };

  const findPersonCardAtPoint = (x: number, y: number): string | null => {
    const el = document.elementFromPoint(x, y);
    const card = el?.closest<HTMLElement>('[data-person-id]');
    return card?.dataset.personId || null;
  };

  const handleGripPointerDown = (e: React.PointerEvent<HTMLButtonElement>, personId: string) => {
    if (busy || editingId === personId || persons.length < 2) return;
    e.stopPropagation();
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // ignore if capture is unavailable
    }
    pointerDragRef.current = {
      sourceId: personId,
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      active: false,
    };
  };

  const handleGripPointerMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const state = pointerDragRef.current;
    if (!state || state.pointerId !== e.pointerId) return;
    const dx = e.clientX - state.startX;
    const dy = e.clientY - state.startY;
    if (!state.active && Math.hypot(dx, dy) < 6) return;

    if (!state.active) {
      state.active = true;
      setDragSourceId(state.sourceId);
    }

    setPointerGhost({ x: e.clientX, y: e.clientY });

    // Gentle edge auto-scroll when dragging near viewport top/bottom on mobile
    if (e.clientY < 72) {
      window.scrollBy(0, -14);
    } else if (e.clientY > window.innerHeight - 72) {
      window.scrollBy(0, 14);
    }

    const hitId = findPersonCardAtPoint(e.clientX, e.clientY);
    setDropTargetId(hitId && hitId !== state.sourceId ? hitId : null);
  };

  const handleGripPointerEnd = (e: React.PointerEvent<HTMLButtonElement>) => {
    const state = pointerDragRef.current;
    if (!state || state.pointerId !== e.pointerId) return;
    try {
      if (e.currentTarget.hasPointerCapture(e.pointerId)) {
        e.currentTarget.releasePointerCapture(e.pointerId);
      }
    } catch {
      // ignore
    }

    const wasActive = state.active;
    const sourceId = state.sourceId;
    const hitId = wasActive ? findPersonCardAtPoint(e.clientX, e.clientY) : null;
    resetDragState();

    if (wasActive) {
      justDraggedUntilRef.current = Date.now() + 260;
      if (hitId && hitId !== sourceId) {
        openMergeConfirm(sourceId, hitId);
      }
    }
  };

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
    if (selectedIds.length === 2) {
      openMergeConfirm(selectedIds[1], selectedIds[0]);
      return;
    }
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

  const handleRecluster = async () => {
    setBusy(true);
    try {
      const res = await api.reclusterPersons();
      onToast(
        `Reagrupado con alta precisión: ${res.persons_before} → ${res.persons_after} persona(s) (${res.reassigned_count} reasignadas). Nombres guardados conservados.`
      );
      onRefresh();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al reagrupar');
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
            id="btn-recluster-people"
            className="btn"
            disabled={busy || persons.length === 0}
            onClick={handleRecluster}
            title="Vuelve a agrupar todas las capturas con máxima precisión sin perder tus nombres guardados"
          >
            <RefreshCw size={15} /> Reagrupar
          </button>

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
        {persons.length >= 2 && (
          <span
            className="studio-strip-item"
            style={{ marginLeft: 'auto', color: 'var(--ink-secondary)', fontSize: '0.8rem' }}
          >
            <GitMerge size={14} style={{ color: 'var(--accent)' }} />
            Arrastra una persona sobre otra para indicar que son la misma
          </span>
        )}
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
          const imgSrc = getPersonAvatarSrc(person, previewMode);
          const canDrag = editingId !== person.id && !busy && filtered.length >= 2;
          const isDragging = dragSourceId === person.id;
          const isDropTarget = dropTargetId === person.id && dragSourceId !== person.id;

          return (
            <article
              key={person.id}
              data-person-id={person.id}
              draggable={canDrag}
              onDragStart={(e) => {
                if (!canDrag) {
                  e.preventDefault();
                  return;
                }
                setDragSourceId(person.id);
                setDropTargetId(null);
                e.dataTransfer.effectAllowed = 'move';
                e.dataTransfer.setData('text/plain', person.id);
              }}
              onDragOver={(e) => {
                const sourceId = dragSourceId;
                if (!sourceId || sourceId === person.id) return;
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                if (dropTargetId !== person.id) {
                  setDropTargetId(person.id);
                }
              }}
              onDragLeave={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
                  if (dropTargetId === person.id) {
                    setDropTargetId(null);
                  }
                }
              }}
              onDrop={(e) => {
                e.preventDefault();
                const sourceId = dragSourceId || e.dataTransfer.getData('text/plain');
                resetDragState();
                if (sourceId && sourceId !== person.id) {
                  justDraggedUntilRef.current = Date.now() + 260;
                  openMergeConfirm(sourceId, person.id);
                }
              }}
              onDragEnd={() => {
                justDraggedUntilRef.current = Date.now() + 200;
                resetDragState();
              }}
              className={`person-card ${isSelected ? 'selected' : ''} ${
                isDragging ? 'is-dragging' : ''
              } ${isDropTarget ? 'is-drop-target' : ''}`}
            >
              <div
                className={`person-card-media ${showCutout ? 'checkerboard-bg cutout-preview' : ''}`}
                onClick={() => {
                  if (Date.now() < justDraggedUntilRef.current) return;
                  onSelectPerson(person.id);
                }}
              >
                {imgSrc ? (
                  <img src={imgSrc} alt={person.name} loading="lazy" draggable={false} />
                ) : (
                  <div
                    style={{
                      width: '100%',
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: 8,
                      background: 'var(--bg-darkroom)',
                      color: 'var(--ink-inverse)',
                    }}
                  >
                    <span
                      style={{
                        fontFamily: 'var(--font-serif)',
                        fontSize: '2.4rem',
                        fontWeight: 600,
                        letterSpacing: '-0.03em',
                      }}
                    >
                      {getPersonInitials(person.name)}
                    </span>
                    <span className="badge badge-dark" style={{ fontSize: '0.72rem' }}>
                      Solo identidad (sin fotos)
                    </span>
                  </div>
                )}

                {isDropTarget && (
                  <div className="drop-target-overlay">
                    <div className="drop-target-pill">
                      <GitMerge size={20} />
                      <strong>¿Misma persona?</strong>
                      <span>Soltar para unir con {person.name}</span>
                    </div>
                  </div>
                )}

                <div className="card-overlay-top">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
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

                    {filtered.length >= 2 && (
                      <button
                        type="button"
                        draggable={false}
                        className="person-drag-handle"
                        aria-label={`Arrastrar a ${person.name} sobre otra persona para unir`}
                        title="Arrastra sobre otra persona para indicar que son la misma"
                        onClick={(e) => e.stopPropagation()}
                        onDragStart={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                        }}
                        onPointerDown={(e) => handleGripPointerDown(e, person.id)}
                        onPointerMove={handleGripPointerMove}
                        onPointerUp={handleGripPointerEnd}
                        onPointerCancel={handleGripPointerEnd}
                      >
                        <GripVertical size={14} />
                      </button>
                    )}
                  </div>

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
                  if (Date.now() < justDraggedUntilRef.current) return;
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
                        setNameDraft(isDefaultPersonName(person.name) ? '' : person.name);
                      }}
                    >
                      <Edit3 size={13} />
                      {isDefaultPersonName(person.name) ? 'Nombrar' : ''}
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

      {pointerGhost && dragSourceId && (() => {
        const draggingPerson = persons.find((p) => p.id === dragSourceId);
        const hoveredPerson = dropTargetId ? persons.find((p) => p.id === dropTargetId) : null;
        if (!draggingPerson) return null;
        const thumb = getPersonAvatarSrc(draggingPerson, 'portrait');
        return (
          <div
            className="drag-ghost-chip"
            style={{
              left: pointerGhost.x,
              top: pointerGhost.y,
            }}
          >
            {thumb ? (
              <img src={thumb} alt={draggingPerson.name} className="drag-ghost-avatar" />
            ) : (
              <span className="drag-ghost-initials">
                {getPersonInitials(draggingPerson.name)}
              </span>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2 }}>
              <strong style={{ fontSize: '0.84rem' }}>{draggingPerson.name}</strong>
              <span style={{ fontSize: '0.73rem', opacity: 0.85 }}>
                {hoveredPerson
                  ? `Soltar → unir con ${hoveredPerson.name}`
                  : 'Arrastra sobre otra persona'}
              </span>
            </div>
          </div>
        );
      })()}

      {pendingMerge && (
        <div
          className="modal-backdrop"
          onClick={() => {
            if (!busy) setPendingMerge(null);
          }}
        >
          <div
            className="modal-card"
            style={{ maxWidth: 480 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
                <GitMerge size={18} style={{ color: 'var(--accent)' }} />
                <h2 style={{ margin: 0, fontSize: '1.12rem', fontWeight: 700 }}>
                  ¿Son la misma persona?
                </h2>
              </div>
              <button
                className="btn btn-ghost btn-sm"
                aria-label="Cerrar"
                disabled={busy}
                onClick={() => setPendingMerge(null)}
              >
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <div className="merge-preview-row">
                {[pendingMerge.source, pendingMerge.target].map((p, idx) => {
                  const avatar = getPersonAvatarSrc(p, 'portrait');
                  return (
                    <div key={p.id} style={{ display: 'contents' }}>
                      {idx === 1 && (
                        <div className="merge-preview-arrow" aria-hidden="true">
                          <ArrowRight size={18} />
                        </div>
                      )}
                      <div className="merge-preview-card">
                        <div className="merge-preview-avatar">
                          {avatar ? (
                            <img src={avatar} alt={p.name} />
                          ) : (
                            <span>{getPersonInitials(p.name)}</span>
                          )}
                        </div>
                        <strong className="merge-preview-name">{p.name}</strong>
                        <span className="merge-preview-meta">
                          {p.total_detections > 0
                            ? `${p.total_detections} captura(s)`
                            : 'Solo identidad'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="field">
                <label htmlFor="input-merge-final-name">Nombre final de la persona</label>
                <input
                  id="input-merge-final-name"
                  className="input"
                  value={mergeNameDraft}
                  placeholder="Nombre del amigo..."
                  autoFocus
                  onChange={(e) => setMergeNameDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') void handleConfirmDragMerge();
                    if (e.key === 'Escape' && !busy) setPendingMerge(null);
                  }}
                />
                <span className="field-hint">
                  Se unirán todas sus fotos, recortes PNG, apariciones en vídeos y su perfil facial
                  para futuros reconocimientos automáticos.
                </span>
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn"
                disabled={busy}
                onClick={() => setPendingMerge(null)}
              >
                Cancelar
              </button>
              <button
                id="btn-confirm-drag-merge"
                className="btn btn-primary"
                disabled={busy}
                onClick={() => void handleConfirmDragMerge()}
              >
                <GitMerge size={15} />
                {busy ? 'Uniendo...' : 'Sí, son la misma persona'}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
