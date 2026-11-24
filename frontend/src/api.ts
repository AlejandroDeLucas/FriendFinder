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

export function getApiBase(): string {
  const saved = window.localStorage.getItem(API_BASE_STORAGE_KEY)?.trim();
  if (saved) return saved.replace(/\/+$/, '');
  const envBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  return envBase ? envBase.replace(/\/+$/, '') : '';
}

export function setApiBase(url: string): void {
  const cleaned = url.trim().replace(/\/+$/, '');
  if (cleaned) {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, cleaned);
  } else {
    window.localStorage.removeItem(API_BASE_STORAGE_KEY);
  }
}

export function getStoredPin(): string {
  return window.sessionStorage.getItem(PIN_STORAGE_KEY) || '';
}

export function setStoredPin(pin: string): void {
  if (pin.trim()) {
    window.sessionStorage.setItem(PIN_STORAGE_KEY, pin.trim());
  } else {
    window.sessionStorage.removeItem(PIN_STORAGE_KEY);
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const pin = getStoredPin();
  if (pin) headers.set('X-Access-Pin', pin);

  const res = await fetch(withBase(path), { ...init, headers });
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) {
        detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // ignore non-json error body
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const assetUrl = {
  portrait: (name?: string | null) => (name ? withPinQuery(withBase(`/assets/portraits/${name}`)) : ''),
  cutout: (name?: string | null) => (name ? withPinQuery(withBase(`/assets/cutouts/${name}`)) : ''),
  crop: (name?: string | null) => (name ? withPinQuery(withBase(`/assets/crops/${name}`)) : ''),
  frame: (name?: string | null) => (name ? withPinQuery(withBase(`/assets/frames/${name}`)) : ''),
  thumb: (name?: string | null) => (name ? withPinQuery(withBase(`/assets/thumbs/${name}`)) : ''),
  mediaStream: (mediaId: string) => withPinQuery(withBase(`/api/media/${mediaId}/stream`)),
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
  getAuthStatus: () => request<{ pin_required: boolean }>('/api/auth/status'),

  verifyPin: (pin: string) =>
    request<{ valid: boolean }>('/api/auth/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pin }),
    }),

  getStats: () => request<SystemStats>('/api/stats'),

  getSettings: () =>
    request<{ settings: AppSettings; device: SystemStats['device'] }>('/api/settings'),

  updateSettings: (payload: Record<string, unknown>) =>
    request<{ settings: AppSettings; device: SystemStats['device'] }>('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  listPersons: (includeIgnored = false) =>
    request<PersonSummary[]>(`/api/persons?include_ignored=${includeIgnored}`),

  getPerson: (personId: string) => request<PersonDetail>(`/api/persons/${personId}`),

  updatePerson: (
    personId: string,
    payload: { name?: string; notes?: string; cover_detection_id?: string; is_ignored?: boolean }
  ) =>
    request<PersonSummary>(`/api/persons/${personId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  mergePersons: (targetId: string, sourceIds: string[]) =>
    request<PersonSummary>(`/api/persons/${targetId}/merge`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_person_ids: sourceIds }),
    }),

  deletePerson: (personId: string) =>
    request<{ deleted: boolean }>(`/api/persons/${personId}`, { method: 'DELETE' }),

  analyzePersonScenes: (personId: string, backend?: string) =>
    request<{ analyzed_count: number; backend: string }>(
      `/api/persons/${personId}/analyze-scene`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: backend || null }),
      }
    ),

  exportZip: async (payload: {
    person_ids?: string[] | null;
    include_portraits?: boolean;
    include_cutouts?: boolean;
    include_crops?: boolean;
    include_originals?: boolean;
    only_kept?: boolean;
  }) => {
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

  listMedia: (mediaType?: 'image' | 'video') =>
    request<MediaItem[]>(mediaType ? `/api/media?media_type=${mediaType}` : '/api/media'),

  getMediaDetail: (mediaId: string) => request<MediaItem>(`/api/media/${mediaId}`),

  setMediaRetention: (mediaId: string, keepForever: boolean) =>
    request<MediaItem>(`/api/media/${mediaId}/retention`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep_forever: keepForever }),
    }),

  cleanupExpiredMedia: () =>
    request<{ purged_count: number; freed_bytes: number }>('/api/media/cleanup-expired', {
      method: 'POST',
    }),

  deleteMedia: (mediaId: string, keepPersonAssets = true) =>
    request<{ deleted: boolean; message: string }>(
      `/api/media/${mediaId}?keep_person_assets=${keepPersonAssets}`,
      { method: 'DELETE' }
    ),

  getJob: (jobId: string) => request<JobInfo>(`/api/jobs/${jobId}`),

  listJobs: () => request<JobInfo[]>('/api/jobs'),

  updateDetectionStatus: (detId: string, status: 'kept' | 'discarded', discardReason?: string) =>
    request<Detection>(`/api/detections/${detId}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, discard_reason: discardReason }),
    }),

  reassignDetection: (detId: string, payload: { person_id?: string; new_person_name?: string }) =>
    request<Detection>(`/api/detections/${detId}/reassign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  analyzeDetectionScene: (detId: string, backend?: string) =>
    request<Detection>(`/api/detections/${detId}/analyze-scene`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: backend || null }),
    }),

  deleteDetection: (detId: string) =>
    request<{ deleted: boolean }>(`/api/detections/${detId}`, { method: 'DELETE' }),

  searchByImage: async (file: File, faceIndex = 0, threshold?: number) => {
    const form = new FormData();
    form.append('file', file);
    form.append('face_index', String(faceIndex));
    if (threshold !== undefined) form.append('threshold', String(threshold));
    return request<SearchResult>('/api/search/by-image', { method: 'POST', body: form });
  },
};
