import { useEffect, useState } from 'react';
import { Cpu, Save, X } from 'lucide-react';
import { api } from '../api';
import type { AppSettings, SystemStats } from '../types';

interface Props {
  onClose: () => void;
  onSaved: (settings: AppSettings) => void;
  onToast: (msg: string) => void;
}

export function SettingsModal({ onClose, onSaved, onToast }: Props) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [device, setDevice] = useState<SystemStats['device'] | null>(null);
  const [saving, setSaving] = useState(false);

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
    if (!settings) return;
    setSaving(true);
    try {
      const res = await api.updateSettings({
        video_sample_fps: Number(settings.video_sample_fps),
        similarity_threshold: Number(settings.similarity_threshold),
        min_quality_score: Number(settings.min_quality_score),
        max_portraits_per_person: Number(settings.max_portraits_per_person),
        max_fullbody_per_person: Number(settings.max_fullbody_per_person),
        min_video_gap_seconds: Number(settings.min_video_gap_seconds),
        auto_scene_analysis: settings.auto_scene_analysis === 'true',
        scene_backend: settings.scene_backend,
        ollama_url: settings.ollama_url,
        ollama_model: settings.ollama_model,
      });
      onSaved(res.settings);
      onToast('Ajustes locales guardados');
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
              Ajustes locales de FriendFinder
            </h2>
            <div style={{ fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
              100% local y privado — tus fotos y vídeos nunca salen de tu equipo
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
              <Cpu size={17} style={{ color: 'var(--accent)' }} />
              <span>
                Aceleración detectada: <strong className="mono">{device.torch_device.toUpperCase()}</strong> ·
                ONNX <strong className="mono">{device.onnx_provider}</strong>
              </span>
            </div>
          )}

          {settings && (
            <>
              <div className="field-grid">
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
                  <span className="field-hint">0.45 por defecto. Sube si mezcla personas parecidas.</span>
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
                  <span className="field-hint">Capturas por debajo se descartan automáticamente.</span>
                </div>

                <div className="field">
                  <label>Separación mínima entre frames de vídeo (seg)</label>
                  <input
                    className="input"
                    type="number"
                    step="0.5"
                    min="0"
                    max="20"
                    value={settings.min_video_gap_seconds}
                    onChange={(e) =>
                      setSettings({ ...settings, min_video_gap_seconds: e.target.value })
                    }
                  />
                  <span className="field-hint">Evita guardar 10 frames casi idénticos seguidos.</span>
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
                  <label>Máximo mejores cuerpos enteros / PNG por persona</label>
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
                      <option value="ollama">Ollama local (llava / moondream / llama3.2-vision)</option>
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
        </div>

        <div className="modal-footer">
          <button className="btn" onClick={onClose}>
            Cancelar
          </button>
          <button
            id="btn-save-settings"
            className="btn btn-primary"
            disabled={saving || !settings}
            onClick={handleSave}
          >
            <Save size={15} /> {saving ? 'Guardando...' : 'Guardar ajustes'}
          </button>
        </div>
      </div>
    </div>
  );
}
