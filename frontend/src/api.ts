import {
  clearBrowserPersonPhotos,
  deleteBrowserMedia,
  deleteBrowserPerson,
  deleteStoredDetection,
  ensureBlobsHydrated,
  exportBrowserBackup,
  getBrowserMediaDetail,
  getBrowserPersonDetail,
  getBrowserSettings,
  getCachedBlobUrl,
  getStoredDetection,
  importBrowserBackup,
  listBrowserMedia,
  listBrowserPersons,
  mergeBrowserPersons,
  purgeExpiredBrowserSources,
  setBrowserMediaRetention,
  updateBrowserPerson,
  updateBrowserSettings,
} from './browserEngine/db';
import {
  exportZipBrowser,
  getBrowserJob,
  getBrowserStats,
  listBrowserJobs,
  reassignBrowserDetection,
  reclusterAllBrowserPersons,
  searchByImageBrowser,
  updateBrowserDetectionStatus,
  uploadMediaBrowser,
} from './browserEngine/pipeline';
import type {
  AppSettings,
  Detection,
  JobInfo,
  MediaItem,
  PersonDetail,
  PersonSummary,
  SearchResult,
  SystemStats,
} from './types';

const API_BASE_STORAGE_KEY = 'friendfinder_api_base';
const PIN_STORAGE_KEY = 'friendfinder_access_pin';

export type EngineMode = 'server' | 'browser';
let currentEngineMode: EngineMode = 'server';
let modeDetectedPromise: Promise<EngineMode> | null = null;
let reaperStarted = false;

export function getEngineMode(): EngineMode {
  return currentEngineMode;
}

export function getApiBase(): string {
  try {
    const saved = window.localStorage.getItem(API_BASE_STORAGE_KEY)?.trim();
    if (saved) return saved.replace(/\/+$/, '');
  } catch {
    // ignore storage SecurityError
  }
  const envBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  return envBase ? envBase.replace(/\/+$/, '') : '';
}

export function setApiBase(url: string): void {
  const cleaned = url.trim().replace(/\/+$/, '');
  try {
    if (cleaned) {
      window.localStorage.setItem(API_BASE_STORAGE_KEY, cleaned);
    } else {
      window.localStorage.removeItem(API_BASE_STORAGE_KEY);
    }
  } catch {
    // ignore storage errors
  }
}

export function getStoredPin(): string {
  try {
    return window.sessionStorage.getItem(PIN_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

export function setStoredPin(pin: string): void {
  try {
    if (pin.trim()) {
      window.sessionStorage.setItem(PIN_STORAGE_KEY, pin.trim());
    } else {
      window.sessionStorage.removeItem(PIN_STORAGE_KEY);
    }
  } catch {
    // ignore storage errors
  }
}

function withBase(path: string): string {
  const base = getApiBase();
  return `${base}${path}`;
}

function withPinQuery(url: string): string {
  const pin = getStoredPin();
  if (!pin || !url) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}pin=${encodeURIComponent(pin)}`;
}

/** Auto-detect whether a Python FastAPI backend is reachable; otherwise activate 100% autonomous browser AI mode. */
export function initEngineMode(): Promise<EngineMode> {
  if (modeDetectedPromise) return modeDetectedPromise;
  modeDetectedPromise = (async () => {
    try {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 2200);
      const res = await fetch(withBase('/api/health'), { signal: controller.signal });
      window.clearTimeout(timer);
      const ct = res.headers.get('content-type') || '';
      if (res.ok && ct.includes('application/json')) {
        const data = await res.json();
        if (data?.status === 'ok') {
          currentEngineMode = 'server';
          return 'server';
        }
      }
    } catch {
      // No backend reachable -> autonomous browser mode
    }

    currentEngineMode = 'browser';
    await ensureBlobsHydrated();
    await purgeExpiredBrowserSources();
    if (!reaperStarted) {
      reaperStarted = true;
      window.setInterval(() => {
        void purgeExpiredBrowserSources();
      }, 30000);
    }
    return 'browser';
  })();
  return modeDetectedPromise;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const pin = getStoredPin();
  if (pin) headers.set('X-Access-Pin', pin);

  const res = await fetch(withBase(path), { ...init, headers });
  const contentType = res.headers.get('content-type') || '';

  if (!res.ok) {
    let detail = `Error ${res.status}`;
    if (contentType.includes('application/json')) {
      try {
        const data = await res.json();
        if (data?.detail) {
          detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        }
      } catch {
        // ignore
      }
    }
    throw new Error(detail);
  }

  if (!contentType.includes('application/json')) {
    throw new Error('Backend no conectado (la respuesta no es JSON)');
  }
  return res.json() as Promise<T>;
}

export const assetUrl = {
  portrait: (name?: string | null) => {
    if (!name) return '';
    return currentEngineMode === 'browser'
      ? getCachedBlobUrl(name)
      : withPinQuery(withBase(`/assets/portraits/${name}`));
  },
  cutout: (name?: string | null) => {
    if (!name) return '';
    return currentEngineMode === 'browser'
      ? getCachedBlobUrl(name)
      : withPinQuery(withBase(`/assets/cutouts/${name}`));
  },
  crop: (name?: string | null) => {
    if (!name) return '';
    return currentEngineMode === 'browser'
      ? getCachedBlobUrl(name)
      : withPinQuery(withBase(`/assets/crops/${name}`));
  },
  frame: (name?: string | null) => {
    if (!name) return '';
    return currentEngineMode === 'browser'
      ? getCachedBlobUrl(name)
      : withPinQuery(withBase(`/assets/frames/${name}`));
  },
  thumb: (name?: string | null) => {
    if (!name) return '';
    return currentEngineMode === 'browser'
      ? getCachedBlobUrl(name)
      : withPinQuery(withBase(`/assets/thumbs/${name}`));
  },
  mediaStream: (mediaId: string) =>
    currentEngineMode === 'browser'
      ? getCachedBlobUrl(`upload_${mediaId}`)
      : withPinQuery(withBase(`/api/media/${mediaId}/stream`)),
};

export function formatTime(sec: number): string {
  const s = Math.max(0, sec);
  const mins = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(mins).padStart(2, '0')}:${rem < 10 ? '0' : ''}${rem.toFixed(1)}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function formatTimeRemaining(expiresAt?: string | null): string | null {
  if (!expiresAt) return null;
  const diffMs = new Date(expiresAt).getTime() - Date.now();
  if (Number.isNaN(diffMs)) return null;
  if (diffMs <= 0) return 'Expirando...';
  const mins = Math.ceil(diffMs / 60000);
  if (mins < 60) return `${mins} min`;
  const hours = Math.floor(mins / 60);
  const rem = mins % 60;
  return rem > 0 ? `${hours}h ${rem}m` : `${hours}h`;
}

export const api = {
  getAuthStatus: async (): Promise<{ pin_required: boolean }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return { pin_required: false };
    return request<{ pin_required: boolean }>('/api/auth/status');
  },

  verifyPin: async (pin: string): Promise<{ valid: boolean }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return { valid: true };
    return request<{ valid: boolean }>('/api/auth/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pin }),
    });
  },

  getStats: async (): Promise<SystemStats> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return getBrowserStats();
    return request<SystemStats>('/api/stats');
  },

  getSettings: async (): Promise<{ settings: AppSettings; device: SystemStats['device'] }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') {
      const settings = await getBrowserSettings();
      return {
        settings,
        device: {
          torch_device: 'Navegador (WebGL/WASM)',
          onnx_provider: '100% local en tu móvil/PC',
          onnx_providers: 'browser',
        },
      };
    }
    return request<{ settings: AppSettings; device: SystemStats['device'] }>('/api/settings');
  },

  updateSettings: async (
    payload: Record<string, unknown>
  ): Promise<{ settings: AppSettings; device: SystemStats['device'] }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') {
      const settings = await updateBrowserSettings(payload);
      return {
        settings,
        device: {
          torch_device: 'Navegador (WebGL/WASM)',
          onnx_provider: '100% local en tu móvil/PC',
          onnx_providers: 'browser',
        },
      };
    }
    return request<{ settings: AppSettings; device: SystemStats['device'] }>('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  listPersons: async (includeIgnored = false): Promise<PersonSummary[]> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return listBrowserPersons(includeIgnored);
    return request<PersonSummary[]>(`/api/persons?include_ignored=${includeIgnored}`);
  },

  getPerson: async (personId: string): Promise<PersonDetail> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return getBrowserPersonDetail(personId);
    return request<PersonDetail>(`/api/persons/${personId}`);
  },

  updatePerson: async (
    personId: string,
    payload: {
      name?: string;
      notes?: string;
      tags?: string[];
      cover_detection_id?: string;
      is_ignored?: boolean;
    }
  ): Promise<PersonSummary> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return updateBrowserPerson(personId, payload);
    return request<PersonSummary>(`/api/persons/${personId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  mergePersons: async (targetId: string, sourceIds: string[]): Promise<PersonSummary> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return mergeBrowserPersons(targetId, sourceIds);
    return request<PersonSummary>(`/api/persons/${targetId}/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_person_ids: sourceIds }),
    });
  },

  deletePerson: async (personId: string): Promise<{ deleted: boolean }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return deleteBrowserPerson(personId);
    return request<{ deleted: boolean }>(`/api/persons/${personId}`, { method: 'DELETE' });
  },

  clearPersonPhotos: async (
    personId: string
  ): Promise<{ cleared_detections: number; person_preserved: boolean }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return clearBrowserPersonPhotos(personId);
    return request(`/api/persons/${personId}/clear-photos`, { method: 'POST' });
  },

  reclusterPersons: async (): Promise<{
    persons_before: number;
    persons_after: number;
    reassigned_count: number;
  }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return reclusterAllBrowserPersons();
    return request('/api/persons/recluster', { method: 'POST' });
  },

  analyzePersonScenes: async (
    personId: string,
    backend?: string
  ): Promise<{ analyzed_count: number; backend: string }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') {
      const detail = await getBrowserPersonDetail(personId);
      return { analyzed_count: detail.kept_detections.length, backend: 'browser-ai' };
    }
    return request<{ analyzed_count: number; backend: string }>(
      `/api/persons/${personId}/analyze-scene`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backend || null }),
      }
    );
  },

  exportZip: async (payload: {
    person_ids?: string[] | null;
    include_portraits?: boolean;
    include_cutouts?: boolean;
    include_crops?: boolean;
    include_originals?: boolean;
    only_kept?: boolean;
  }): Promise<void> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return exportZipBrowser(payload);

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    const pin = getStoredPin();
    if (pin) headers['X-Access-Pin'] = pin;

    const res = await fetch(withBase('/api/persons/export'), {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Error generando el paquete ZIP');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'FriendFinder_GameAssets.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  },

  uploadMedia: async (files: File[], sampleFps?: number, keepForever = false) => {
    const mode = await initEngineMode();
    if (mode === 'browser') return uploadMediaBrowser(files, sampleFps, keepForever);

    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    if (sampleFps !== undefined) form.append('sample_fps', String(sampleFps));
    form.append('keep_forever', keepForever ? 'true' : 'false');
    return request<{
      job_id: string;
      media_ids: string[];
      uploaded_count: number;
      keep_forever: boolean;
      skipped: Array<{ filename: string; reason: string }>;
    }>('/api/media/upload', { method: 'POST', body: form });
  },

  listMedia: async (mediaType?: 'image' | 'video'): Promise<MediaItem[]> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return listBrowserMedia(mediaType);
    return request<MediaItem[]>(mediaType ? `/api/media?media_type=${mediaType}` : '/api/media');
  },

  getMediaDetail: async (mediaId: string): Promise<MediaItem> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return getBrowserMediaDetail(mediaId);
    return request<MediaItem>(`/api/media/${mediaId}`);
  },

  setMediaRetention: async (mediaId: string, keepForever: boolean): Promise<MediaItem> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return setBrowserMediaRetention(mediaId, keepForever);
    return request<MediaItem>(`/api/media/${mediaId}/retention`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep_forever: keepForever }),
    });
  },

  cleanupExpiredMedia: async (): Promise<{ purged_count: number; freed_bytes: number }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return purgeExpiredBrowserSources();
    return request<{ purged_count: number; freed_bytes: number }>('/api/media/cleanup-expired', {
      method: 'POST',
    });
  },

  deleteMedia: async (
    mediaId: string,
    keepPersonAssets = true
  ): Promise<{ deleted: boolean; message: string }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return deleteBrowserMedia(mediaId, keepPersonAssets);
    return request<{ deleted: boolean; message: string }>(
      `/api/media/${mediaId}?keep_person_assets=${keepPersonAssets}`,
      { method: 'DELETE' }
    );
  },

  getJob: async (jobId: string): Promise<JobInfo> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return getBrowserJob(jobId);
    return request<JobInfo>(`/api/jobs/${jobId}`);
  },

  listJobs: async (): Promise<JobInfo[]> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return listBrowserJobs();
    return request<JobInfo[]>('/api/jobs');
  },

  updateDetectionStatus: async (
    detId: string,
    status: 'kept' | 'discarded',
    discardReason?: string
  ): Promise<Detection> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return updateBrowserDetectionStatus(detId, status, discardReason);
    return request<Detection>(`/api/detections/${detId}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, discard_reason: discardReason }),
    });
  },

  reassignDetection: async (
    detId: string,
    payload: { person_id?: string; new_person_name?: string }
  ): Promise<Detection> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return reassignBrowserDetection(detId, payload);
    return request<Detection>(`/api/detections/${detId}/reassign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  analyzeDetectionScene: async (detId: string, backend?: string): Promise<Detection> => {
    const mode = await initEngineMode();
    if (mode === 'browser') {
      const det = await getStoredDetection(detId);
      if (!det) throw new Error('Detección no encontrada');
      const { embedding: _e, ...clean } = det;
      return clean;
    }
    return request<Detection>(`/api/detections/${detId}/analyze-scene`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: backend || null }),
    });
  },

  deleteDetection: async (detId: string): Promise<{ deleted: boolean }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') {
      await deleteStoredDetection(detId);
      return { deleted: true };
    }
    return request<{ deleted: boolean }>(`/api/detections/${detId}`, { method: 'DELETE' });
  },

  searchByImage: async (file: File, faceIndex = 0, threshold?: number): Promise<SearchResult> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return searchByImageBrowser(file, faceIndex, threshold);

    const form = new FormData();
    form.append('file', file);
    form.append('face_index', String(faceIndex));
    if (threshold !== undefined) form.append('threshold', String(threshold));
    return request<SearchResult>('/api/search/by-image', { method: 'POST', body: form });
  },

  exportBackup: async (): Promise<Record<string, unknown>> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return exportBrowserBackup();
    return request<Record<string, unknown>>('/api/backup/export');
  },

  importBackup: async (
    payload: Record<string, unknown>
  ): Promise<{ imported_persons: number; updated_persons: number }> => {
    const mode = await initEngineMode();
    if (mode === 'browser') return importBrowserBackup(payload);
    return request<{ imported_persons: number; updated_persons: number }>('/api/backup/import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },
};
