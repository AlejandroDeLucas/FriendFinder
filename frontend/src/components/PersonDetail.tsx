import { useEffect, useState } from 'react';
import {
  ArrowLeft,
  Check,
  Download,
  Edit3,
  Eye,
  Film,
  Image as ImageIcon,
  ImageOff,
  Layers,
  Play,
  RotateCcw,
  Scissors,
  Sparkles,
  Star,
  Trash2,
  UserPlus,
  X,
} from 'lucide-react';
import { api, assetUrl, formatTime } from '../api';
import type { Detection, PersonDetail as PersonDetailType, PersonSummary } from '../types';

interface Props {
  personId: string;
  sceneBackend: 'heuristic' | 'blip' | 'ollama';
  onBack: () => void;
  onOpenMediaAt: (mediaId: string, timestampSec: number) => void;
  onPersonChanged: () => void;
  onToast: (msg: string) => void;
}

type FormatView = 'portrait' | 'cutout' | 'original';

export function PersonDetail({
  personId,
  sceneBackend,
  onBack,
  onOpenMediaAt,
  onPersonChanged,
  onToast,
}: Props) {
  const [person, setPerson] = useState<PersonDetailType | null>(null);
  const [otherPersons, setOtherPersons] = useState<PersonSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [subtab, setSubtab] = useState<'kept' | 'all' | 'discarded' | 'appearances'>('kept');
  const [selectedDet, setSelectedDet] = useState<Detection | null>(null);
  const [formatView, setFormatView] = useState<FormatView>('portrait');
  const [editingName, setEditingName] = useState(false);
  const [nameDraft, setNameDraft] = useState('');
  const [busyAction, setBusyAction] = useState(false);
  const [exportOpts, setExportOpts] = useState({
    portraits: true,
    cutouts: true,
    crops: true,
    originals: false,
  });
  const [showExportPanel, setShowExportPanel] = useState(false);

  const loadPerson = async (keepSelectionId?: string) => {
    setLoading(true);
    try {
      const [data, allPeople] = await Promise.all([
        api.getPerson(personId),
        api.listPersons(),
      ]);
      setPerson(data);
      setOtherPersons(allPeople.filter((p) => p.id !== personId));
      setNameDraft(data.name);

      const all = [...data.kept_detections, ...data.discarded_detections];
      if (keepSelectionId) {
        const found = all.find((d) => d.id === keepSelectionId);
        if (found) {
          setSelectedDet(found);
          return;
        }
      }
      setSelectedDet(data.kept_detections[0] || data.discarded_detections[0] || null);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error cargando persona');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadPerson();
  }, [personId]);

  const handleSaveName = async () => {
    if (!person || !nameDraft.trim()) return;
    try {
      await api.updatePerson(person.id, { name: nameDraft.trim() });
      setEditingName(false);
      onToast(`Renombrado a "${nameDraft.trim()}"`);
      onPersonChanged();
      await loadPerson(selectedDet?.id);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al renombrar');
    }
  };

  const handleSplitToNewPerson = async (det: Detection) => {
    const suggested = window.prompt(
      '¿Qué nombre quieres ponerle a esta nueva persona? (déjalo vacío para asignarlo luego)',
      ''
    );
    if (suggested === null) return;
    setBusyAction(true);
    try {
      await api.reassignDetection(det.id, { new_person_name: suggested.trim() || undefined });
      onToast(
        suggested.trim()
          ? `Separado como "${suggested.trim()}"`
          : 'Separado en un nuevo perfil de persona'
      );
      onPersonChanged();
      await loadPerson();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al separar persona');
    } finally {
      setBusyAction(false);
    }
  };

  const handleMoveToPerson = async (det: Detection, targetId: string) => {
    if (!targetId) return;
    const targetName = otherPersons.find((p) => p.id === targetId)?.name || 'otra persona';
    setBusyAction(true);
    try {
      await api.reassignDetection(det.id, { person_id: targetId });
      onToast(`Captura movida a "${targetName}"`);
      onPersonChanged();
      await loadPerson();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al mover captura');
    } finally {
      setBusyAction(false);
    }
  };

  const handleDeleteDetection = async (det: Detection) => {
    if (
      !window.confirm(
        `¿Eliminar esta captura?\n\nEl perfil y el reconocimiento facial de "${person?.name || 'esta persona'}" se conservarán.`
      )
    ) {
      return;
    }
    setBusyAction(true);
    try {
      await api.deleteDetection(det.id);
      onToast(
        `Captura eliminada. La identidad de "${person?.name || 'esta persona'}" se sigue conservando.`
      );
      onPersonChanged();
      await loadPerson();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error eliminando captura');
    } finally {
      setBusyAction(false);
    }
  };

  const handleClearPhotos = async () => {
    if (!person) return;
    if (
      !window.confirm(
        `¿Borrar TODAS las fotos y recortes de "${person.name}"?\n\nSu perfil y su firma facial SE CONSERVARÁN para reconocerle automáticamente en futuras fotos y vídeos.`
      )
    ) {
      return;
    }
    setBusyAction(true);
    try {
      const res = await api.clearPersonPhotos(person.id);
      onToast(
        `${res.cleared_detections} foto(s) borrada(s). El perfil de "${person.name}" se conserva para reconocerle en futuras subidas.`
      );
      onPersonChanged();
      await loadPerson();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error borrando fotos');
    } finally {
      setBusyAction(false);
    }
  };

  const handleSetCover = async (detId: string) => {
    if (!person) return;
    try {
      await api.updatePerson(person.id, { cover_detection_id: detId });
      onToast('Portada actualizada');
      onPersonChanged();
      await loadPerson(detId);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error actualizando portada');
    }
  };

  const handleToggleStatus = async (det: Detection) => {
    setBusyAction(true);
    try {
      const nextStatus = det.status === 'kept' ? 'discarded' : 'kept';
      await api.updateDetectionStatus(det.id, nextStatus);
      onToast(
        nextStatus === 'kept'
          ? 'Captura rescatada y todos los formatos generados.'
          : 'Captura movida a descartadas.'
      );
      onPersonChanged();
      await loadPerson(det.id);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error actualizando estado');
    } finally {
      setBusyAction(false);
    }
  };

  const handleAnalyzeScene = async (det: Detection, backendOverride?: string) => {
    setBusyAction(true);
    try {
      const updated = await api.analyzeDetectionScene(det.id, backendOverride || sceneBackend);
      setSelectedDet(updated);
      onToast(`Escena analizada (${updated.scene_backend})`);
      await loadPerson(det.id);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error analizando escena');
    } finally {
      setBusyAction(false);
    }
  };

  const handleAnalyzeAllScenes = async () => {
    if (!person) return;
    setBusyAction(true);
    try {
      const res = await api.analyzePersonScenes(person.id, sceneBackend);
      onToast(`${res.analyzed_count} captura(s) analizadas con motor "${res.backend}".`);
      await loadPerson(selectedDet?.id);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error analizando escenas');
    } finally {
      setBusyAction(false);
    }
  };

  const handleExport = async () => {
    if (!person) return;
    setBusyAction(true);
    try {
      await api.exportZip({
        person_ids: [person.id],
        include_portraits: exportOpts.portraits,
        include_cutouts: exportOpts.cutouts,
        include_crops: exportOpts.crops,
        include_originals: exportOpts.originals,
        only_kept: true,
      });
      setShowExportPanel(false);
      onToast(`Pack ZIP de ${person.name} descargado.`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error exportando ZIP');
    } finally {
      setBusyAction(false);
    }
  };

  const handleDeletePerson = async () => {
    if (!person) return;
    if (!window.confirm(`¿Eliminar el perfil "${person.name}" y todas sus detecciones?`)) return;
    try {
      await api.deletePerson(person.id);
      onToast('Perfil eliminado');
      onPersonChanged();
      onBack();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error al eliminar');
    }
  };

  if (loading && !person) {
    return <div className="panel">Cargando ficha de persona...</div>;
  }

  if (!person) {
    return (
      <div className="panel">
        <p>Persona no encontrada.</p>
        <button className="btn" onClick={onBack}>
          Volver
        </button>
      </div>
    );
  }

  const allDetections = [...person.kept_detections, ...person.discarded_detections];
  const activeList =
    subtab === 'kept'
      ? person.kept_detections
      : subtab === 'all'
        ? allDetections
        : person.discarded_detections;

  const getInspectorImage = (det: Detection | null): { url: string; downloadName: string; isCutout: boolean } => {
    if (!det) return { url: '', downloadName: 'image.jpg', isCutout: false };
    if (formatView === 'cutout' && det.cutout_path) {
      return {
        url: assetUrl.cutout(det.cutout_path),
        downloadName: `${person.name}_recorte_${det.id.slice(0, 6)}.png`,
        isCutout: true,
      };
    }
    if (formatView === 'original' && (det.frame_path || det.crop_path)) {
      return {
        url: assetUrl.frame(det.frame_path) || assetUrl.crop(det.crop_path),
        downloadName: `${person.name}_frame_${det.id.slice(0, 6)}.jpg`,
        isCutout: false,
      };
    }
    return {
      url: assetUrl.portrait(det.portrait_path) || assetUrl.thumb(det.thumb_path),
      downloadName: `${person.name}_retrato_${det.id.slice(0, 6)}.jpg`,
      isCutout: false,
    };
  };

  const inspectorAsset = getInspectorImage(selectedDet);

  return (
    <section>
      <div style={{ marginBottom: 14 }}>
        <button id="btn-back-people" className="btn btn-ghost btn-sm" onClick={onBack}>
          <ArrowLeft size={15} /> Volver a personas
        </button>
      </div>

      <header className="page-header">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {editingName ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <input
                id="input-rename-person"
                className="input"
                style={{ width: 260, fontSize: '1.15rem', fontWeight: 600 }}
                value={nameDraft}
                autoFocus
                onChange={(e) => setNameDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void handleSaveName();
                  if (e.key === 'Escape') setEditingName(false);
                }}
              />
              <button className="btn btn-primary btn-sm" onClick={handleSaveName}>
                <Check size={15} /> Guardar
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => setEditingName(false)}>
                <X size={15} />
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h1 className="page-title">{person.name}</h1>
              <button
                id="btn-edit-person-name"
                className="btn btn-ghost btn-sm"
                onClick={() => setEditingName(true)}
                title="Renombrar amigo"
              >
                <Edit3 size={15} /> Renombrar
              </button>
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="badge badge-success">{person.kept_count} mejores capturas</span>
            <span className="badge">
              <Scissors size={12} /> {person.cutout_count} recortes PNG sin fondo
            </span>
            <span className="badge">{person.media_count} archivo(s)</span>
            {person.discarded_count > 0 && (
              <span className="badge badge-warning">{person.discarded_count} descartadas</span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            id="btn-analyze-person-scenes"
            className="btn"
            disabled={busyAction || person.kept_count === 0}
            onClick={handleAnalyzeAllScenes}
            title="Entender qué está haciendo en sus capturas (100% local)"
          >
            <Sparkles size={15} /> Analizar escenas ({sceneBackend})
          </button>

          <button
            id="btn-toggle-export"
            className="btn btn-dark"
            disabled={busyAction || person.kept_count === 0}
            onClick={() => setShowExportPanel((v) => !v)}
          >
            <Download size={15} /> Exportar Pack Videojuego
          </button>

          {person.total_detections > 0 && (
            <button
              id="btn-clear-person-photos"
              className="btn"
              disabled={busyAction}
              onClick={handleClearPhotos}
              title="Borra las fotos y recortes pero conserva la identidad facial para futuras fotos y vídeos"
            >
              <ImageOff size={15} /> Borrar fotos (conservar persona)
            </button>
          )}

          <button className="btn btn-danger btn-icon" onClick={handleDeletePerson} title="Eliminar persona por completo">
            <Trash2 size={15} />
          </button>
        </div>
      </header>

      {person.total_detections === 0 && (
        <div
          className="panel"
          style={{
            marginBottom: 20,
            background: 'var(--success-soft)',
            borderColor: 'hsla(150, 42%, 31%, 0.24)',
          }}
        >
          <strong style={{ color: 'var(--success)' }}>
            Identidad guardada sin fotos (0 imágenes almacenadas)
          </strong>
          <p style={{ margin: '6px 0 0', fontSize: '0.88rem', color: 'var(--ink-primary)' }}>
            El reconocimiento facial de <strong>{person.name}</strong> sigue activo. Aunque hayas
            borrado sus fotos, cuando vuelva a aparecer en una foto o vídeo futuro se le asignará
            automáticamente el nombre <strong>{person.name}</strong>.
          </p>
        </div>
      )}

      {showExportPanel && (
        <div className="panel" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div>
              <strong>Configurar exportación ZIP para {person.name}</strong>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 10 }}>
                <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={exportOpts.portraits}
                    onChange={(e) => setExportOpts((o) => ({ ...o, portraits: e.target.checked }))}
                  />
                  Retratos (cara + hombros)
                </label>
                <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={exportOpts.cutouts}
                    onChange={(e) => setExportOpts((o) => ({ ...o, cutouts: e.target.checked }))}
                  />
                  Recortes PNG sin fondo / sin otras personas
                </label>
                <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={exportOpts.crops}
                    onChange={(e) => setExportOpts((o) => ({ ...o, crops: e.target.checked }))}
                  />
                  Recortes rectangulares limpios
                </label>
                <label style={{ display: 'inline-flex', gap: 6, alignItems: 'center', fontSize: '0.88rem' }}>
                  <input
                    type="checkbox"
                    checked={exportOpts.originals}
                    onChange={(e) => setExportOpts((o) => ({ ...o, originals: e.target.checked }))}
                  />
                  Frames originales completos
                </label>
              </div>
            </div>
            <button className="btn btn-primary" disabled={busyAction} onClick={handleExport}>
              <Download size={15} /> Descargar ZIP
            </button>
          </div>
        </div>
      )}

      {/* Top Spotlight: Best Portrait vs Best Full-Body Cutout */}
      <div className="detail-hero-grid">
        <div className="spotlight-card">
          <div
            className="spotlight-media"
            onClick={() => {
              if (person.best_portrait) {
                setSelectedDet(person.best_portrait);
                setFormatView('portrait');
              }
            }}
          >
            {person.best_portrait ? (
              <img
                src={
                  assetUrl.portrait(person.best_portrait.portrait_path) ||
                  assetUrl.thumb(person.best_portrait.thumb_path)
                }
                alt="Mejor retrato"
              />
            ) : (
              <ImageIcon size={28} style={{ color: 'var(--ink-muted)' }} />
            )}
          </div>
          <div className="spotlight-info">
            <div>
              <span className="badge badge-accent">Mejor Retrato</span>
              <h2 className="section-title" style={{ marginTop: 8, marginBottom: 4 }}>
                {person.best_portrait
                  ? `Calidad ${person.best_portrait.portrait_score.toFixed(0)}/100`
                  : 'Sin retrato'}
              </h2>
              {person.best_portrait && (
                <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-secondary)' }}>
                  Nitidez {person.best_portrait.sharpness_score.toFixed(0)} · Frontalidad{' '}
                  {person.best_portrait.frontality_score.toFixed(0)}
                </p>
              )}
            </div>
            {person.best_portrait?.portrait_path && (
              <a
                className="btn btn-sm"
                href={assetUrl.portrait(person.best_portrait.portrait_path)}
                download={`${person.name}_mejor_retrato.jpg`}
              >
                <Download size={14} /> Descargar retrato
              </a>
            )}
          </div>
        </div>

        <div className="spotlight-card">
          <div
            className={`spotlight-media ${person.best_body?.cutout_path ? 'checkerboard-bg cutout' : ''}`}
            onClick={() => {
              if (person.best_body) {
                setSelectedDet(person.best_body);
                setFormatView(person.best_body.cutout_path ? 'cutout' : 'original');
              }
            }}
          >
            {person.best_body?.cutout_path ? (
              <img src={assetUrl.cutout(person.best_body.cutout_path)} alt="Mejor recorte PNG" />
            ) : person.best_body?.crop_path ? (
              <img src={assetUrl.crop(person.best_body.crop_path)} alt="Mejor cuerpo" />
            ) : (
              <Scissors size={28} style={{ color: 'var(--ink-muted)' }} />
            )}
          </div>
          <div className="spotlight-info">
            <div>
              <span className="badge badge-success">Mejor Recorte PNG (sin otros)</span>
              <h2 className="section-title" style={{ marginTop: 8, marginBottom: 4 }}>
                {person.best_body
                  ? `Cuerpo ${person.best_body.body_score.toFixed(0)}/100`
                  : 'Sin recorte de cuerpo'}
              </h2>
              {person.best_body ? (
                <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-secondary)' }}>
                  Separación de otras personas: {person.best_body.separation_score.toFixed(0)}%{' '}
                  {person.best_body.is_full_body ? '· Cuerpo entero' : '· Medio cuerpo'}
                </p>
              ) : (
                <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink-muted)' }}>
                  Sube fotos o vídeos donde salga de cuerpo entero para extraer su silueta PNG.
                </p>
              )}
            </div>
            {person.best_body?.cutout_path && (
              <a
                className="btn btn-sm"
                href={assetUrl.cutout(person.best_body.cutout_path)}
                download={`${person.name}_mejor_recorte.png`}
              >
                <Download size={14} /> Descargar PNG transparente
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Subtabs */}
      <nav className="subtabs">
        <button
          id="subtab-kept"
          className={`subtab ${subtab === 'kept' ? 'active' : ''}`}
          onClick={() => {
            setSubtab('kept');
            if (person.kept_detections[0]) setSelectedDet(person.kept_detections[0]);
          }}
        >
          <Star size={15} /> Mejores ({person.kept_detections.length})
        </button>
        <button
          id="subtab-all"
          className={`subtab ${subtab === 'all' ? 'active' : ''}`}
          onClick={() => {
            setSubtab('all');
            if (allDetections[0]) setSelectedDet(allDetections[0]);
          }}
        >
          <Layers size={15} /> Todas ({allDetections.length})
        </button>
        <button
          id="subtab-discarded"
          className={`subtab ${subtab === 'discarded' ? 'active' : ''}`}
          onClick={() => {
            setSubtab('discarded');
            if (person.discarded_detections[0]) setSelectedDet(person.discarded_detections[0]);
          }}
        >
          <RotateCcw size={15} /> Descartadas ({person.discarded_detections.length})
        </button>
        <button
          id="subtab-appearances"
          className={`subtab ${subtab === 'appearances' ? 'active' : ''}`}
          onClick={() => setSubtab('appearances')}
        >
          <Film size={15} /> Fotos y vídeos donde aparece ({person.media_appearances.length})
        </button>
      </nav>

      {subtab === 'appearances' ? (
        <div className="result-list">
          {person.media_appearances.map((m) => (
            <div key={m.media_id} className="result-row" style={{ alignItems: 'flex-start' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  {m.media_type === 'video' ? (
                    <Film size={16} style={{ color: 'var(--accent)' }} />
                  ) : (
                    <ImageIcon size={16} style={{ color: 'var(--success)' }} />
                  )}
                  <strong>{m.filename}</strong>
                  <span className="badge">{m.appearances_count} aparición(es)</span>
                  {m.source_deleted && (
                    <span className="badge badge-warning">Original borrado (capturas conservadas)</span>
                  )}
                </div>

                <div className="timestamp-chips" style={{ marginTop: 10 }}>
                  {m.timestamps.slice(0, 20).map((ts) => (
                    <button
                      key={ts.detection_id}
                      className="ts-chip"
                      disabled={m.source_deleted}
                      onClick={() => onOpenMediaAt(m.media_id, ts.timestamp_sec)}
                    >
                      {m.media_type === 'video' ? (
                        <>
                          <Play size={11} /> {formatTime(ts.timestamp_sec)}
                        </>
                      ) : (
                        <>
                          <Eye size={11} /> Abrir foto
                        </>
                      )}
                      <span style={{ color: 'var(--ink-muted)' }}>
                        ★{ts.portrait_score.toFixed(0)}
                      </span>
                    </button>
                  ))}
                  {m.timestamps.length > 20 && (
                    <span className="badge">+{m.timestamps.length - 20} más</span>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="captures-layout">
          <div>
            {activeList.length === 0 ? (
              <div className="panel" style={{ color: 'var(--ink-secondary)' }}>
                {subtab === 'kept'
                  ? 'No hay capturas seleccionadas todavía.'
                  : 'No hay capturas descartadas para esta persona.'}
              </div>
            ) : (
              <div className="captures-grid">
                {activeList.map((det) => {
                  const thumb =
                    assetUrl.portrait(det.portrait_path) || assetUrl.thumb(det.thumb_path);
                  const active = selectedDet?.id === det.id;
                  return (
                    <article
                      key={det.id}
                      className={`capture-tile ${active ? 'active' : ''}`}
                      onClick={() => {
                        setSelectedDet(det);
                        if (formatView === 'cutout' && !det.cutout_path) {
                          setFormatView('portrait');
                        }
                      }}
                    >
                      <div className="capture-tile-img">
                        {thumb && <img src={thumb} alt="Captura" loading="lazy" />}
                        <div className="card-overlay-top">
                          <span className="badge badge-dark mono">
                            ★ {det.overall_score.toFixed(0)}
                          </span>
                          {det.has_cutout && (
                            <span className="badge badge-dark">
                              <Scissors size={10} /> PNG
                            </span>
                          )}
                        </div>
                        {det.media_type === 'video' && (
                          <div className="card-overlay-bottom">
                            <span className="badge badge-dark mono">
                              {formatTime(det.timestamp_sec)}
                            </span>
                          </div>
                        )}
                      </div>

                      <div className="capture-tile-foot">
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>Retrato {det.portrait_score.toFixed(0)}</span>
                          {det.body_score > 0 && <span>Cuerpo {det.body_score.toFixed(0)}</span>}
                        </div>
                        {det.discard_reason && (
                          <div
                            style={{
                              fontSize: '0.72rem',
                              color: 'var(--warning)',
                              lineHeight: 1.3,
                            }}
                          >
                            {det.discard_reason}
                          </div>
                        )}
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </div>

          {/* Right Sticky 3-Format Inspector */}
          {selectedDet && (
            <aside className="inspector-panel">
              <div className="format-switcher">
                <button
                  id="fmt-portrait"
                  className={`format-btn ${formatView === 'portrait' ? 'active' : ''}`}
                  onClick={() => setFormatView('portrait')}
                >
                  1. Retrato
                </button>
                <button
                  id="fmt-cutout"
                  className={`format-btn ${formatView === 'cutout' ? 'active' : ''}`}
                  disabled={!selectedDet.cutout_path}
                  onClick={() => setFormatView('cutout')}
                  title={
                    selectedDet.cutout_path
                      ? 'PNG transparente sin otras personas ni fondo'
                      : 'Rescata esta captura o elige una con cuerpo segmentado'
                  }
                >
                  2. Recorte PNG
                </button>
                <button
                  id="fmt-original"
                  className={`format-btn ${formatView === 'original' ? 'active' : ''}`}
                  disabled={!selectedDet.frame_path && !selectedDet.crop_path}
                  onClick={() => setFormatView('original')}
                >
                  3. Frame original
                </button>
              </div>

              <div className={`inspector-stage ${inspectorAsset.isCutout ? 'cutout-mode' : ''}`}>
                {inspectorAsset.url ? (
                  <img src={inspectorAsset.url} alt="Vista previa de captura" />
                ) : (
                  <span style={{ color: 'var(--ink-muted)' }}>Sin vista previa</span>
                )}
              </div>

              <div className="inspector-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <div style={{ fontSize: '0.84rem', color: 'var(--ink-secondary)' }}>
                    <strong>{selectedDet.media_filename}</strong>
                    {selectedDet.media_type === 'video' && (
                      <span className="mono"> · {formatTime(selectedDet.timestamp_sec)}</span>
                    )}
                  </div>
                  {!selectedDet.source_deleted && (
                    <button
                      className="ts-chip"
                      onClick={() =>
                        onOpenMediaAt(selectedDet.media_id, selectedDet.timestamp_sec)
                      }
                    >
                      <Play size={11} /> Ver fuente
                    </button>
                  )}
                </div>

                {selectedDet.discard_reason && (
                  <div
                    style={{
                      padding: '9px 12px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--warning-soft)',
                      color: 'var(--warning)',
                      fontSize: '0.83rem',
                      fontWeight: 500,
                    }}
                  >
                    {selectedDet.discard_reason}
                  </div>
                )}

                <div className="score-bars">
                  {[
                    ['Retrato', selectedDet.portrait_score],
                    ['Cuerpo / Silueta', selectedDet.body_score],
                    ['Nitidez', selectedDet.sharpness_score],
                    ['Frontalidad', selectedDet.frontality_score],
                  ].map(([label, val]) => (
                    <div key={String(label)} className="score-metric">
                      <div className="score-metric-top">
                        <span>{label}</span>
                        <span className="mono">{Number(val).toFixed(0)}</span>
                      </div>
                      <div className="score-bar-bg">
                        <div
                          className={`score-bar-fill ${Number(val) >= 72 ? 'high' : ''}`}
                          style={{ width: `${Math.min(100, Number(val))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Scene / Activity Understanding Box */}
                <div
                  style={{
                    padding: '11px 13px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-canvas)',
                    border: '1px solid var(--border-hairline)',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 4,
                    }}
                  >
                    <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--ink-secondary)' }}>
                      Qué está haciendo (análisis local)
                    </span>
                    <button
                      className="btn btn-ghost btn-sm"
                      style={{ padding: '2px 7px', fontSize: '0.75rem' }}
                      disabled={busyAction}
                      onClick={() => handleAnalyzeScene(selectedDet)}
                    >
                      <Sparkles size={12} /> Reanalizar
                    </button>
                  </div>
                  <div style={{ fontSize: '0.86rem' }}>
                    {selectedDet.scene_summary || 'Sin descripción todavía.'}
                  </div>
                  {selectedDet.scene_tags && selectedDet.scene_tags.length > 0 && (
                    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 7 }}>
                      {selectedDet.scene_tags.map((tag) => (
                        <span key={tag} className="badge" style={{ fontSize: '0.7rem' }}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                  {inspectorAsset.url && (
                    <a
                      className="btn btn-primary btn-sm"
                      href={inspectorAsset.url}
                      download={inspectorAsset.downloadName}
                    >
                      <Download size={14} /> Descargar vista actual
                    </a>
                  )}
                  {selectedDet.crop_path && (
                    <a
                      className="btn btn-sm"
                      href={assetUrl.crop(selectedDet.crop_path)}
                      download={`${person.name}_recorte_rect_${selectedDet.id.slice(0, 6)}.jpg`}
                    >
                      Recorte limpio
                    </a>
                  )}
                  {selectedDet.status === 'kept' && (
                    <button
                      className="btn btn-sm"
                      onClick={() => handleSetCover(selectedDet.id)}
                    >
                      Usar de portada
                    </button>
                  )}
                  <button
                    id="btn-toggle-rescue"
                    className={`btn btn-sm ${selectedDet.status === 'discarded' ? 'btn-dark' : ''}`}
                    disabled={busyAction}
                    onClick={() => handleToggleStatus(selectedDet)}
                  >
                    {selectedDet.status === 'discarded' ? (
                      <>
                        <RotateCcw size={14} /> Rescatar y generar formatos
                      </>
                    ) : (
                      'Descartar'
                    )}
                  </button>
                </div>

                {/* Split / Reassign Misgrouped Capture */}
                <div
                  style={{
                    borderTop: '1px solid var(--border-hairline)',
                    paddingTop: 12,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8,
                  }}
                >
                  <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--ink-secondary)' }}>
                    ¿No es {person.name}? Corregir agrupación:
                  </span>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                    <button
                      id="btn-split-person"
                      className="btn btn-sm"
                      disabled={busyAction}
                      onClick={() => handleSplitToNewPerson(selectedDet)}
                    >
                      <UserPlus size={14} /> Separar en nueva persona
                    </button>

                    {otherPersons.length > 0 && (
                      <select
                        className="select"
                        style={{ width: 'auto', padding: '5px 9px', fontSize: '0.8rem' }}
                        value=""
                        disabled={busyAction}
                        onChange={(e) => {
                          if (e.target.value) void handleMoveToPerson(selectedDet, e.target.value);
                        }}
                      >
                        <option value="">Mover a...</option>
                        {otherPersons.map((op) => (
                          <option key={op.id} value={op.id}>
                            {op.name}
                          </option>
                        ))}
                      </select>
                    )}

                    <button
                      className="btn btn-danger btn-sm"
                      style={{ padding: '6px 9px' }}
                      disabled={busyAction}
                      onClick={() => handleDeleteDetection(selectedDet)}
                      title="Eliminar esta captura"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              </div>
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
