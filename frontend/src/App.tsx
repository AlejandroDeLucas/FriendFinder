import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Camera,
  Film,
  Lock,
  Play,
  Search,
  Settings,
  Upload,
  Users,
  X,
} from 'lucide-react';
import {
  type EngineMode,
  api,
  assetUrl,
  formatTime,
  getEngineMode,
  getStoredPin,
  initEngineMode,
  setStoredPin,
} from './api';
import { MediaLibrary } from './components/MediaLibrary';
import { PeopleGallery } from './components/PeopleGallery';
import { PersonDetail } from './components/PersonDetail';
import { SearchByPhoto } from './components/SearchByPhoto';
import { SettingsModal } from './components/SettingsModal';
import { UploadStudio } from './components/UploadStudio';
import type { AppSettings, JobInfo, MediaItem, PersonSummary, SystemStats } from './types';

type Tab = 'people' | 'upload' | 'search' | 'library';

export function App() {
  const [tab, setTab] = useState<Tab>('people');
  const [selectedPersonId, setSelectedPersonId] = useState<string | null>(null);
  const [persons, setPersons] = useState<PersonSummary[]>([]);
  const [media, setMedia] = useState<MediaItem[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [engineMode, setEngineMode] = useState<EngineMode>('browser');
  const [showSettings, setShowSettings] = useState(false);

  // Optional Cloudflare PIN access gate
  const [pinRequired, setPinRequired] = useState(false);
  const [pinUnlocked, setPinUnlocked] = useState(true);
  const [pinInput, setPinInput] = useState('');
  const [pinError, setPinError] = useState('');

  // Active background job polling
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<JobInfo | null>(null);
  const [lastFinishedJob, setLastFinishedJob] = useState<JobInfo | null>(null);

  // Global Media / Video Player Modal (with timestamp seeking!)
  const [viewerMedia, setViewerMedia] = useState<MediaItem | null>(null);
  const [viewerSeekSec, setViewerSeekSec] = useState<number>(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Toast notification
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<number | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 3800);
  }, []);

  const refreshAll = useCallback(async () => {
    try {
      const mode = await initEngineMode();
      setEngineMode(mode);
      const [pList, mList, sData, setRes] = await Promise.all([
        api.listPersons(),
        api.listMedia(),
        api.getStats(),
        api.getSettings(),
      ]);
      setPersons(Array.isArray(pList) ? pList : []);
      setMedia(Array.isArray(mList) ? mList : []);
      setStats(sData);
      setSettings(setRes.settings);
      if (!activeJobId && Array.isArray(sData?.active_jobs) && sData.active_jobs.length > 0) {
        setActiveJobId(sData.active_jobs[0].id);
        setActiveJob(sData.active_jobs[0]);
      }
    } catch {
      setEngineMode(getEngineMode());
    } finally {
      setLoading(false);
    }
  }, [activeJobId]);

  // Check optional PIN lock first
  useEffect(() => {
    let mounted = true;
    api
      .getAuthStatus()
      .then(async (status) => {
        if (!mounted) return;
        if (!status.pin_required) {
          setPinRequired(false);
          setPinUnlocked(true);
          void refreshAll();
          return;
        }
        setPinRequired(true);
        const existing = getStoredPin();
        if (existing) {
          try {
            await api.verifyPin(existing);
            if (!mounted) return;
            setPinUnlocked(true);
            void refreshAll();
            return;
          } catch {
            setStoredPin('');
          }
        }
        setPinUnlocked(false);
        setLoading(false);
      })
      .catch(() => {
        if (mounted) void refreshAll();
      });
    return () => {
      mounted = false;
    };
  }, [refreshAll]);

  const handlePinSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPinError('');
    try {
      await api.verifyPin(pinInput);
      setStoredPin(pinInput);
      setPinUnlocked(true);
      setPinInput('');
      void refreshAll();
    } catch {
      setPinError('PIN incorrecto. Inténtalo de nuevo.');
    }
  };

  // Poll active background job
  useEffect(() => {
    if (!activeJobId) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const job = await api.getJob(activeJobId);
        if (cancelled) return;
        setActiveJob(job);
        if (job.status === 'done') {
          setLastFinishedJob(job);
          setActiveJobId(null);
          showToast(
            `Análisis terminado: ${job.summary?.kept_detections ?? 0} mejores capturas extraídas.`
          );
          void refreshAll();
        } else if (job.status === 'error') {
          setActiveJobId(null);
          showToast(`Error en el análisis: ${job.error || 'Error desconocido'}`);
          void refreshAll();
        }
      } catch {
        // ignore transient polling error
      }
    };

    void poll();
    const interval = window.setInterval(poll, 1100);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [activeJobId, refreshAll, showToast]);

  const openMediaAt = async (mediaId: string, timestampSec = 0) => {
    try {
      const detail = await api.getMediaDetail(mediaId);
      setViewerSeekSec(timestampSec);
      setViewerMedia(detail);
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'No se pudo abrir el archivo');
    }
  };

  useEffect(() => {
    if (viewerMedia?.media_type === 'video' && videoRef.current) {
      const el = videoRef.current;
      const onLoaded = () => {
        if (viewerSeekSec > 0) {
          el.currentTime = viewerSeekSec;
        }
      };
      el.addEventListener('loadedmetadata', onLoaded);
      if (el.readyState >= 1 && viewerSeekSec > 0) {
        el.currentTime = viewerSeekSec;
      }
      return () => el.removeEventListener('loadedmetadata', onLoaded);
    }
  }, [viewerMedia, viewerSeekSec]);

  const seekVideoTo = (sec: number) => {
    setViewerSeekSec(sec);
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      void videoRef.current.play().catch(() => undefined);
    }
  };

  const navigateTab = (next: Tab) => {
    setSelectedPersonId(null);
    setTab(next);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (pinRequired && !pinUnlocked) {
    return (
      <div className="pin-gate">
        <form className="pin-card" onSubmit={handlePinSubmit}>
          <div className="empty-studio-icon" style={{ marginBottom: 4 }}>
            <Lock size={24} />
          </div>
          <h1 className="section-title" style={{ margin: 0, fontSize: '1.45rem' }}>
            FriendFinder protegido
          </h1>
          <p style={{ margin: 0, fontSize: '0.88rem', color: 'var(--ink-secondary)' }}>
            Introduce el PIN configurado en tu servidor para acceder desde este dispositivo.
          </p>
          <input
            id="input-access-pin"
            className="input"
            type="password"
            inputMode="numeric"
            placeholder="PIN de acceso"
            value={pinInput}
            autoFocus
            onChange={(e) => setPinInput(e.target.value)}
            style={{ textAlign: 'center', fontSize: '1.1rem', letterSpacing: '0.18em' }}
          />
          {pinError && (
            <div style={{ color: 'var(--danger)', fontSize: '0.83rem', fontWeight: 600 }}>
              {pinError}
            </div>
          )}
          <button id="btn-unlock-pin" type="submit" className="btn btn-primary" disabled={!pinInput.trim()}>
            Entrar
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand" onClick={() => navigateTab('people')}>
            <span className="brand-mark">
              <Camera size={18} />
            </span>
            <span>
              <span className="brand-name">FriendFinder</span>
              <span className="brand-sub">Estudio de personajes</span>
            </span>
          </button>

          <nav className="nav-tabs" aria-label="Navegación principal">
            <button
              id="nav-people"
              className={`nav-tab ${tab === 'people' && !selectedPersonId ? 'active' : ''}`}
              onClick={() => navigateTab('people')}
            >
              <Users size={15} />
              Personas
              <span className="nav-count">{stats?.persons_count ?? persons.length}</span>
            </button>

            <button
              id="nav-upload"
              className={`nav-tab ${tab === 'upload' && !selectedPersonId ? 'active' : ''}`}
              onClick={() => navigateTab('upload')}
            >
              <Upload size={15} />
              Subir y filtrar
            </button>

            <button
              id="nav-search"
              className={`nav-tab ${tab === 'search' && !selectedPersonId ? 'active' : ''}`}
              onClick={() => navigateTab('search')}
            >
              <Search size={15} />
              Buscar por foto
            </button>

            <button
              id="nav-library"
              className={`nav-tab ${tab === 'library' && !selectedPersonId ? 'active' : ''}`}
              onClick={() => navigateTab('library')}
            >
              <Film size={15} />
              Biblioteca
              <span className="nav-count">{media.length}</span>
            </button>
          </nav>

          <div className="topbar-actions">
            <span
              className={`badge ${engineMode === 'server' ? 'badge-success' : 'badge-accent'}`}
              title={
                engineMode === 'server'
                  ? 'Conectado al backend Python (InsightFace + YOLOv8-seg)'
                  : 'IA ejecutándose 100% en tu navegador (sin servidor)'
              }
            >
              {engineMode === 'server' ? 'Servidor IA' : 'IA local web'}
            </span>
            <button
              id="btn-open-settings"
              className="btn btn-icon"
              onClick={() => setShowSettings(true)}
              title="Ajustes"
            >
              <Settings size={17} />
            </button>
          </div>
        </div>
      </header>

      {activeJob && (activeJob.status === 'queued' || activeJob.status === 'running') && (
        <div className="job-banner">
          <div className="job-banner-inner">
            <span style={{ fontSize: '0.86rem', fontWeight: 600, flex: 1 }}>
              {activeJob.current_step}
            </span>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.max(4, Math.round(activeJob.progress * 100))}%` }}
              />
            </div>
            <span className="mono" style={{ fontSize: '0.82rem' }}>
              {Math.round(activeJob.progress * 100)}%
            </span>
          </div>
        </div>
      )}

      <main className="main-container">
        {selectedPersonId ? (
          <PersonDetail
            personId={selectedPersonId}
            sceneBackend={settings?.scene_backend || 'heuristic'}
            onBack={() => setSelectedPersonId(null)}
            onOpenMediaAt={openMediaAt}
            onPersonChanged={refreshAll}
            onToast={showToast}
          />
        ) : tab === 'people' ? (
          <PeopleGallery
            persons={persons}
            loading={loading}
            onSelectPerson={(id) => setSelectedPersonId(id)}
            onGoUpload={() => navigateTab('upload')}
            onRefresh={refreshAll}
            onToast={showToast}
          />
        ) : tab === 'upload' ? (
          <UploadStudio
            activeJob={activeJob}
            lastFinishedJob={lastFinishedJob}
            defaultSampleFps={Number(settings?.video_sample_fps || '1.0')}
            retentionMinutes={Number(settings?.source_retention_minutes || '60')}
            onJobStarted={(jobId) => setActiveJobId(jobId)}
            onGoPeople={() => navigateTab('people')}
            onToast={showToast}
          />
        ) : tab === 'search' ? (
          <SearchByPhoto
            onSelectPerson={(id) => setSelectedPersonId(id)}
            onOpenMediaAt={openMediaAt}
            onToast={showToast}
          />
        ) : (
          <MediaLibrary
            media={media}
            onOpenMediaAt={openMediaAt}
            onSelectPerson={(id) => setSelectedPersonId(id)}
            onGoUpload={() => navigateTab('upload')}
            onRefresh={refreshAll}
            onToast={showToast}
          />
        )}
      </main>

      {/* Mobile-First Frosted Bottom Dock */}
      <nav className="mobile-dock" aria-label="Navegación móvil">
        <button
          className={`mobile-dock-btn ${tab === 'people' || selectedPersonId ? 'active' : ''}`}
          onClick={() => navigateTab('people')}
        >
          <Users size={19} />
          <span>Personas</span>
        </button>
        <button
          className={`mobile-dock-btn ${tab === 'upload' && !selectedPersonId ? 'active' : ''}`}
          onClick={() => navigateTab('upload')}
        >
          <Upload size={19} />
          <span>Subir</span>
        </button>
        <button
          className={`mobile-dock-btn ${tab === 'search' && !selectedPersonId ? 'active' : ''}`}
          onClick={() => navigateTab('search')}
        >
          <Search size={19} />
          <span>Buscar</span>
        </button>
        <button
          className={`mobile-dock-btn ${tab === 'library' && !selectedPersonId ? 'active' : ''}`}
          onClick={() => navigateTab('library')}
        >
          <Film size={19} />
          <span>Biblioteca</span>
        </button>
      </nav>

      {/* Media / Video Player Modal with clickable person timestamps */}
      {viewerMedia && (
        <div className="modal-backdrop" onClick={() => setViewerMedia(null)}>
          <div
            className="modal-card"
            style={{ maxWidth: 920 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <h2 className="section-title" style={{ margin: 0 }}>
                  {viewerMedia.original_filename}
                </h2>
                <div style={{ fontSize: '0.82rem', color: 'var(--ink-secondary)' }}>
                  {viewerMedia.verdict_reason}
                </div>
              </div>
              <button className="btn btn-ghost btn-icon" onClick={() => setViewerMedia(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="modal-body">
              <div
                style={{
                  background: 'var(--bg-darkroom)',
                  borderRadius: 'var(--radius-md)',
                  overflow: 'hidden',
                  display: 'grid',
                  placeItems: 'center',
                }}
              >
                {viewerMedia.source_deleted ? (
                  <div style={{ padding: 48, color: 'var(--ink-inverse)', textAlign: 'center' }}>
                    El archivo original expiró o fue borrado, pero las capturas extraídas de las
                    personas siguen guardadas en sus perfiles.
                  </div>
                ) : viewerMedia.media_type === 'video' ? (
                  <video
                    ref={videoRef}
                    src={assetUrl.mediaStream(viewerMedia.id)}
                    controls
                    playsInline
                    style={{ width: '100%', maxHeight: 460 }}
                  />
                ) : (
                  <img
                    src={assetUrl.mediaStream(viewerMedia.id)}
                    alt={viewerMedia.original_filename}
                    style={{ maxHeight: 480, width: 'auto', objectFit: 'contain' }}
                  />
                )}
              </div>

              {viewerMedia.detections && viewerMedia.detections.length > 0 && (
                <div>
                  <h3 style={{ margin: '0 0 10px', fontSize: '0.92rem' }}>
                    Personas detectadas (toca para saltar al instante):
                  </h3>
                  <div className="timestamp-chips">
                    {viewerMedia.detections.slice(0, 40).map((det) => (
                      <button
                        key={det.id}
                        className="ts-chip"
                        onClick={() => {
                          if (viewerMedia.media_type === 'video') {
                            seekVideoTo(det.timestamp_sec);
                          } else if (det.person_id) {
                            setViewerMedia(null);
                            setSelectedPersonId(det.person_id);
                          }
                        }}
                      >
                        {viewerMedia.media_type === 'video' && <Play size={11} />}
                        <span style={{ fontFamily: 'var(--font-sans)', fontWeight: 600 }}>
                          {det.person_name || 'Persona'}
                        </span>
                        {viewerMedia.media_type === 'video' && (
                          <span>{formatTime(det.timestamp_sec)}</span>
                        )}
                        <span style={{ color: 'var(--ink-muted)' }}>
                          ★{det.portrait_score.toFixed(0)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showSettings && (
        <SettingsModal
          onClose={() => setShowSettings(false)}
          onSaved={(s) => setSettings(s)}
          onToast={showToast}
        />
      )}

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}

export default App;
