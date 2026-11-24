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

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `Error ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch {
      // ignore non-json error body
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const assetUrl = {
  portrait: (name?: string | null) => (name ? `/assets/portraits/${name}` : ''),
  cutout: (name?: string | null) => (name ? `/assets/cutouts/${name}` : ''),
  crop: (name?: string | null) => (name ? `/assets/crops/${name}` : ''),
  frame: (name?: string | null) => (name ? `/assets/frames/${name}` : ''),
  thumb: (name?: string | null) => (name ? `/assets/thumbs/${name}` : ''),
  mediaStream: (mediaId: string) => `/api/media/${mediaId}/stream`,
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

export const api = {
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
    const res = await fetch('/api/persons/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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

  uploadMedia: async (files: File[], sampleFps?: number) => {
    const form = new FormData();
    files.forEach((f) => form.append('files', f));
    if (sampleFps !== undefined) form.append('sample_fps', String(sampleFps));
    return request<{
      job_id: string;
      media_ids: string[];
      uploaded_count: number;
      skipped: Array<{ filename: string; reason: string }>;
    }>('/api/media/upload', { method: 'POST', body: form });
  },

  listMedia: (mediaType?: 'image' | 'video') =>
    request<MediaItem[]>(mediaType ? `/api/media?media_type=${mediaType}` : '/api/media'),

  getMediaDetail: (mediaId: string) => request<MediaItem>(`/api/media/${mediaId}`),

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
