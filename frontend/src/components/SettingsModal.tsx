import { useEffect, useRef, useState } from 'react';
import { Cloud, Cpu, Database, Download, Save, Upload, X } from 'lucide-react';
import { api, getApiBase, setApiBase } from '../api';
import type { AppSettings, SystemStats } from '../types';

interface Props {
  onClose: () => void;
  onSaved: (settings: AppSettings) => void;
  onBackupImported?: () => void;
  onToast: (msg: string) => void;
}

export function SettingsModal({ onClose, onSaved, onBackupImported, onToast }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [device, setDevice] = useState<SystemStats['device'] | null>(null);
  const [apiBaseDraft, setApiBaseDraft] = useState<string>(getApiBase());
  const [saving, setSaving] = useState(false);
  const [backupBusy, setBackupBusy] = useState(false);
  const backupInputRef = useRef<HTMLInputElement | null>(null);

  const handleExportBackup = async () => {
    setBackupBusy(true);
    try {
      const data = await api.exportBackup();
      const jsonStr = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonStr], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const day = new Date().toISOString().slice(0, 10);
      a.href = url;
      a.download = `FriendFinder_Backup_${day}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1200);
      const count = Array.isArray(data.persons) ? data.persons.length : 0;
      onToast(`Copia exportada (${count} identidad(es) facial(es) y etiquetas).`);
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error exportando copia de seguridad');
    } finally {
      setBackupBusy(false);
    }
  };

  const handleImportBackupFile = async (file: File) => {
    setBackupBusy(true);
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Record<string, unknown>;
      const res = await api.importBackup(parsed);
      onToast(
        `Copia importada: ${res.imported_persons} persona(s) nueva(s) y ${res.updated_persons} actualizada(s).`
      );
      onBackupImported?.();
    } catch (err) {
      onToast(
        err instanceof Error
          ? `Archivo de copia no válido: ${err.message}`
          : 'Error importando copia de seguridad'
      );
    } finally {
      setBackupBusy(false);
    }
  };

  useEffect(() => {
    api
      .getSettings()
      .then((res) => {
        setSettings(res.settings);
        setDevice(res.device);
      })
      .catch(() => onToast('Error cargando ajustes'));
  }, []);

  const handleSave = async () => {
    const previousBase = getApiBase();
    setApiBase(apiBaseDraft);

    if (!settings) {
      if (apiBaseDraft.trim() !== previousBase) {
        onToast('URL de servidor guardada. Recargando...');
        window.location.reload();
      }
      onClose();
      return;
    }

    setSaving(true);
    try {
      const res = await api.updateSettings({
        video_sample_fps: Number(settings.video_sample_fps),
        similarity_threshold: Number(settings.similarity_threshold),
        min_quality_score: Number(settings.min_quality_score),
        max_portraits_per_person: Number(settings.max_portraits_per_person),
        max_fullbody_per_person: Number(settings.max_fullbody_per_person),
        min_video_gap_seconds: Number(settings.min_video_gap_seconds),
        source_retention_minutes: Number(settings.source_retention_minutes || '60'),
        auto_scene_analysis: settings.auto_scene_analysis === 'true',
        scene_backend: settings.scene_backend,
        ollama_url: settings.ollama_url,
        ollama_model: settings.ollama_model,
      });
      onSaved(res.settings);
      onToast('Ajustes guardados');
      if (apiBaseDraft.trim().replace(/\/+$/, '') !== previousBase) {
        window.location.reload();
        return;
      }
      onClose();
    } catch (err) {
      onToast(err instanceof Error ? err.message : 'Error guardando ajustes');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2 className="section-title" style={{ margin: 0 }}>
              Ajustes de FriendFinder
            </h2>
            <div style={{ fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
              Procesamiento local/privado · Acceso móvil vía Cloudflare Tunnel o Pages
            </div>
          </div>
          <button className="btn btn-ghost btn-icon" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {device && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 14px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-canvas)',
                border: '1px solid var(--border-hairline)',
                fontSize: '0.84rem',
              }}
            >
              <Cpu size={17} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <span>
                Aceleración detectada:{' '}
                <strong className="mono">{device.torch_device.toUpperCase()}</strong> · ONNX{' '}
                <strong className="mono">{device.onnx_provider}</strong>
              </span>
            </div>
          )}

          {settings && (
            <>
              <div className="field-grid">
                <div className="field">
                  <label>Auto-borrado de archivos subidos (minutos)</label>
                  <input
                    className="input"
                    type="number"
                    min="5"
                    max="10080"
                    value={settings.source_retention_minutes || '60'}
                    onChange={(e) =>
                      setSettings({ ...settings, source_retention_minutes: e.target.value })
                    }
                  />
                  <span className="field-hint">
                    60 min (1 hora) por defecto. Solo borra el original subido, nunca los recortes
                    generados.
                  </span>
                </div>

                <div className="field">
                  <label>Frames por segundo en vídeos (FPS)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.5"
                    min="0.2"
                    max="10"
                    value={settings.video_sample_fps}
                    onChange={(e) =>
                      setSettings({ ...settings, video_sample_fps: e.target.value })
                    }
                  />
                  <span className="field-hint">1.0 = 1 frame por segundo (equilibrio ideal)</span>
                </div>

                <div className="field">
                  <label>Umbral de similitud facial (0.30 – 0.70)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.02"
                    min="0.25"
                    max="0.8"
                    value={settings.similarity_threshold}
                    onChange={(e) =>
                      setSettings({ ...settings, similarity_threshold: e.target.value })
                    }
                  />
                  <span className="field-hint">
                    0.45 por defecto. Sube si mezcla personas parecidas.
                  </span>
                </div>

                <div className="field">
                  <label>Calidad mínima para aceptar (0 – 100)</label>
                  <input
                    className="input"
                    type="number"
                    min="0"
                    max="90"
                    value={settings.min_quality_score}
                    onChange={(e) =>
                      setSettings({ ...settings, min_quality_score: e.target.value })
                    }
                  />
                  <span className="field-hint">
                    Capturas por debajo se descartan automáticamente.
                  </span>
                </div>

                <div className="field">
                  <label>Máximo mejores retratos por persona</label>
                  <input
                    className="input"
                    type="number"
                    min="1"
                    max="30"
                    value={settings.max_portraits_per_person}
                    onChange={(e) =>
                      setSettings({ ...settings, max_portraits_per_person: e.target.value })
                    }
                  />
                </div>

                <div className="field">
                  <label>Máximo mejores cuerpos / PNG por persona</label>
                  <input
                    className="input"
                    type="number"
                    min="1"
                    max="30"
                    value={settings.max_fullbody_per_person}
                    onChange={(e) =>
                      setSettings({ ...settings, max_fullbody_per_person: e.target.value })
                    }
                  />
                </div>
              </div>

              <div
                style={{
                  borderTop: '1px solid var(--border-hairline)',
                  paddingTop: 16,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 12,
                }}
              >
                <h3 style={{ margin: 0, fontSize: '0.96rem' }}>
                  Comprensión opcional de escena / actividad (100% local)
                </h3>
                <div className="field-grid">
                  <div className="field">
                    <label>Motor de descripción</label>
                    <select
                      className="select"
                      value={settings.scene_backend}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          scene_backend: e.target.value as AppSettings['scene_backend'],
                        })
                      }
                    >
                      <option value="heuristic">
                        Heurística instantánea (objetos YOLO + plano, 0 descargas)
                      </option>
                      <option value="blip">
                        BLIP local offline (HuggingFace ~990 MB la primera vez)
                      </option>
                      <option value="ollama">
                        Ollama local (llava / moondream / llama3.2-vision)
                      </option>
                    </select>
                  </div>

                  <div className="field">
                    <label>Ejecutar automáticamente al subir</label>
                    <select
                      className="select"
                      value={settings.auto_scene_analysis}
                      onChange={(e) =>
                        setSettings({ ...settings, auto_scene_analysis: e.target.value })
                      }
                    >
                      <option value="false">Solo bajo demanda (más rápido al subir vídeos)</option>
                      <option value="true">Automático en todas las capturas guardadas</option>
                    </select>
                  </div>
                </div>

                {settings.scene_backend === 'ollama' && (
                  <div className="field-grid">
                    <div className="field">
                      <label>URL de Ollama local</label>
                      <input
                        className="input"
                        value={settings.ollama_url}
                        onChange={(e) =>
                          setSettings({ ...settings, ollama_url: e.target.value })
                        }
                      />
                    </div>
                    <div className="field">
                      <label>Modelo de visión en Ollama</label>
                      <input
                        className="input"
                        value={settings.ollama_model}
                        onChange={(e) =>
                          setSettings({ ...settings, ollama_model: e.target.value })
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          {/* Optional Remote Backend URL for Cloudflare Pages mode */}
          <div
            style={{
              borderTop: '1px solid var(--border-hairline)',
              paddingTop: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <h3
              style={{
                margin: 0,
                fontSize: '0.94rem',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 7,
              }}
            >
              <Cloud size={16} style={{ color: 'var(--accent)' }} />
              Conexión Cloudflare (opcional)
            </h3>
            <div className="field">
              <label>URL del backend (solo si alojas el frontend en Cloudflare Pages)</label>
              <input
                className="input"
                placeholder="Dejar vacío para el mismo servidor (ej. https://tu-tunel.trycloudflare.com)"
                value={apiBaseDraft}
                onChange={(e) => setApiBaseDraft(e.target.value)}
              />
              <span className="field-hint">
                Si usas <code>./run-cloudflare.sh</code>, déjalo vacío: todo funciona desde una
                sola URL HTTPS.
              </span>
            </div>
          </div>

          {/* Local Backup Export / Import (.json) */}
          <div
            style={{
              borderTop: '1px solid var(--border-hairline)',
              paddingTop: 16,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
          >
            <h3
              style={{
                margin: 0,
                fontSize: '0.94rem',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 7,
              }}
            >
              <Database size={16} style={{ color: 'var(--accent)' }} />
              Copia de seguridad local (nombres, etiquetas e identidades faciales)
            </h3>
            <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
              Exporta un archivo <code>.json</code> con todas tus personas nombradas, etiquetas y
              perfiles faciales para restaurarlos en otro navegador o móvil sin perder el
              reconocimiento automático.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                id="btn-export-backup"
                type="button"
                className="btn btn-sm"
                disabled={backupBusy}
                onClick={() => void handleExportBackup()}
              >
                <Download size={14} /> Exportar copia (.json)
              </button>
              <button
                id="btn-import-backup"
                type="button"
                className="btn btn-sm"
                disabled={backupBusy}
                onClick={() => backupInputRef.current?.click()}
              >
                <Upload size={14} /> Importar copia (.json)
              </button>
              <input
                ref={backupInputRef}
                type="file"
                accept="application/json,.json"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleImportBackupFile(f);
                  e.target.value = '';
                }}
              />
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn" onClick={onClose}>
            Cancelar
          </button>
          <button
            id="btn-save-settings"
            className="btn btn-primary"
            disabled={saving}
            onClick={handleSave}
          >
            <Save size={15} /> {saving ? 'Guardando...' : 'Guardar ajustes'}
          </button>
        </div>
      </div>
    </div>
  );
}
