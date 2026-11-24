import type {
  AppSettings,
  Detection,
  MediaAppearance,
  MediaItem,
  PersonDetail,
  PersonSummary,
} from '../types';

const DB_NAME = 'friendfinder_browser_v1';
const DB_VERSION = 1;

export interface StoredPerson {
  id: string;
  name: string;
  notes: string;
  centroid: number[] | null;
  reference_embeddings?: number[][];
  cover_detection_id: string | null;
  best_portrait_id: string | null;
  best_body_id: string | null;
  is_ignored: number;
  created_at: string;
  updated_at: string;
}

export interface StoredDetection extends Detection {
  embedding: number[] | null;
}

export interface StoredMedia extends MediaItem {
  upload_blob_key: string | null;
}

export const DEFAULT_BROWSER_SETTINGS: AppSettings = {
  video_sample_fps: '1.0',
  similarity_threshold: '0.45',
  min_quality_score: '32.0',
  max_portraits_per_person: '6',
  max_fullbody_per_person: '4',
  min_video_gap_seconds: '1.5',
  source_retention_minutes: '60',
  auto_scene_analysis: 'false',
  scene_backend: 'heuristic',
  ollama_url: 'http://localhost:11434',
  ollama_model: 'llava',
  yolo_model: 'mediapipe-browser',
  insightface_pack: 'face-api-browser',
};

const blobUrlCache = new Map<string, string>();
let dbPromise: Promise<IDBDatabase> | null = null;
let hydratedBlobs = false;

export function getCachedBlobUrl(key?: string | null): string {
  if (!key) return '';
  return blobUrlCache.get(key) || '';
}

function registerBlobUrl(key: string, blob: Blob): string {
  const existing = blobUrlCache.get(key);
  if (existing) return existing;
  const url = URL.createObjectURL(blob);
  blobUrlCache.set(key, url);
  return url;
}

function revokeBlobUrl(key: string): void {
  const existing = blobUrlCache.get(key);
  if (existing) {
    URL.revokeObjectURL(existing);
    blobUrlCache.delete(key);
  }
}

export function openBrowserDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' });
      }
      if (!db.objectStoreNames.contains('persons')) {
        db.createObjectStore('persons', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('media')) {
        db.createObjectStore('media', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('detections')) {
        db.createObjectStore('detections', { keyPath: 'id' });
      }
      if (!db.objectStoreNames.contains('blobs')) {
        db.createObjectStore('blobs', { keyPath: 'key' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function idbGetAll<T>(storeName: string): Promise<T[]> {
  const db = await openBrowserDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve((req.result || []) as T[]);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet<T>(storeName: string, key: string): Promise<T | undefined> {
  const db = await openBrowserDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).get(key);
    req.onsuccess = () => resolve(req.result as T | undefined);
    req.onerror = () => reject(req.error);
  });
}

async function idbPut<T>(storeName: string, value: T): Promise<void> {
  const db = await openBrowserDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).put(value);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function idbDelete(storeName: string, key: string): Promise<void> {
  const db = await openBrowserDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function ensureBlobsHydrated(): Promise<void> {
  if (hydratedBlobs) return;
  hydratedBlobs = true;
  const rows = await idbGetAll<{ key: string; blob: Blob }>('blobs');
  for (const row of rows) {
    if (row?.key && row?.blob) {
      registerBlobUrl(row.key, row.blob);
    }
  }
}

export function registerSessionBlobUrl(key: string, blob: Blob): string {
  return registerBlobUrl(key, blob);
}

export async function saveBlob(
  key: string,
  blob: Blob,
  options?: { maxPersistBytes?: number }
): Promise<{ url: string; persisted: boolean }> {
  const url = registerBlobUrl(key, blob);
  const limit = options?.maxPersistBytes ?? Infinity;
  if (blob.size > limit) {
    return { url, persisted: false };
  }
  try {
    await idbPut('blobs', { key, blob });
    return { url, persisted: true };
  } catch {
    // Mobile Safari / Chrome IndexedDB quota exceeded: keep working in-memory for this session
    return { url, persisted: false };
  }
}

export async function loadBlob(key: string): Promise<Blob | null> {
  const row = await idbGet<{ key: string; blob: Blob }>('blobs', key);
  return row?.blob || null;
}

export async function deleteBlob(key: string): Promise<void> {
  revokeBlobUrl(key);
  await idbDelete('blobs', key);
}

// ---------- Settings ----------
export async function getBrowserSettings(): Promise<AppSettings> {
  const rows = await idbGetAll<{ key: string; value: string }>('settings');
  const merged: Record<string, string> = { ...DEFAULT_BROWSER_SETTINGS };
  for (const r of rows) {
    if (r.key in merged) merged[r.key] = r.value;
  }
  return merged as unknown as AppSettings;
}

export async function updateBrowserSettings(
  updates: Record<string, unknown>
): Promise<AppSettings> {
  for (const [key, val] of Object.entries(updates)) {
    if (key in DEFAULT_BROWSER_SETTINGS && val !== undefined && val !== null) {
      const strVal = val === true ? 'true' : val === false ? 'false' : String(val);
      await idbPut('settings', { key, value: strVal });
    }
  }
  return getBrowserSettings();
}

// ---------- Media & 1-Hour Auto-Expiration ----------
export async function saveMedia(item: StoredMedia): Promise<void> {
  await idbPut('media', item);
}

export async function getMediaRecord(mediaId: string): Promise<StoredMedia | null> {
  const item = await idbGet<StoredMedia>('media', mediaId);
  return item || null;
}

export async function purgeExpiredBrowserSources(): Promise<{
  purged_count: number;
  freed_bytes: number;
}> {
  const now = Date.now();
  const all = await idbGetAll<StoredMedia>('media');
  let purged = 0;
  let freed = 0;

  for (const m of all) {
    if (
      !m.source_deleted &&
      !m.keep_forever &&
      m.status !== 'processing' &&
      m.expires_at &&
      new Date(m.expires_at).getTime() <= now
    ) {
      if (m.upload_blob_key) {
        await deleteBlob(m.upload_blob_key);
      }
      freed += m.file_size || 0;
      purged += 1;
      await saveMedia({
        ...m,
        source_deleted: 1,
        expires_at: null,
        upload_blob_key: null,
      });
    }
  }
  return { purged_count: purged, freed_bytes: freed };
}

export async function listBrowserMedia(mediaType?: 'image' | 'video'): Promise<MediaItem[]> {
  await ensureBlobsHydrated();
  await purgeExpiredBrowserSources();
  const [allMedia, allDets, allPersons] = await Promise.all([
    idbGetAll<StoredMedia>('media'),
    idbGetAll<StoredDetection>('detections'),
    idbGetAll<StoredPerson>('persons'),
  ]);
  const personNameMap = new Map(allPersons.map((p) => [p.id, p.name]));

  return allMedia
    .filter((m) => (!mediaType ? true : m.media_type === mediaType))
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .map((m) => {
      const seen = new Map<string, string>();
      for (const d of allDets) {
        if (d.media_id === m.id && d.person_id && personNameMap.has(d.person_id)) {
          seen.set(d.person_id, personNameMap.get(d.person_id)!);
        }
      }
      return {
        ...m,
        persons: Array.from(seen.entries()).map(([id, name]) => ({ id, name })),
      };
    });
}

export async function getBrowserMediaDetail(mediaId: string): Promise<MediaItem> {
  await ensureBlobsHydrated();
  const m = await getMediaRecord(mediaId);
  if (!m) throw new Error('Archivo no encontrado');
  const [allDets, allPersons] = await Promise.all([
    idbGetAll<StoredDetection>('detections'),
    idbGetAll<StoredPerson>('persons'),
  ]);
  const personNameMap = new Map(allPersons.map((p) => [p.id, p.name]));
  const detections = allDets
    .filter((d) => d.media_id === mediaId)
    .sort((a, b) => a.timestamp_sec - b.timestamp_sec || b.overall_score - a.overall_score)
    .map((d) => {
      const { embedding: _e, ...rest } = d;
      return {
        ...rest,
        person_name: d.person_id ? personNameMap.get(d.person_id) || 'Persona' : null,
      };
    });
  return { ...m, detections };
}

export async function setBrowserMediaRetention(
  mediaId: string,
  keepForever: boolean
): Promise<MediaItem> {
  const m = await getMediaRecord(mediaId);
  if (!m) throw new Error('Archivo no encontrado');
  if (m.source_deleted) throw new Error('El archivo original ya fue borrado');
  const settings = await getBrowserSettings();
  const mins = Math.max(1, Number(settings.source_retention_minutes || '60'));
  const updated: StoredMedia = {
    ...m,
    keep_forever: keepForever ? 1 : 0,
    expires_at: keepForever ? null : new Date(Date.now() + mins * 60000).toISOString(),
  };
  await saveMedia(updated);
  return updated;
}

export async function deleteBrowserMedia(
  mediaId: string,
  keepPersonAssets = true
): Promise<{ deleted: boolean; message: string }> {
  const m = await getMediaRecord(mediaId);
  if (!m) throw new Error('Archivo no encontrado');
  if (m.upload_blob_key) {
    await deleteBlob(m.upload_blob_key);
  }

  if (keepPersonAssets) {
    await saveMedia({
      ...m,
      source_deleted: 1,
      expires_at: null,
      upload_blob_key: null,
    });
    return {
      deleted: true,
      message:
        'Archivo original eliminado; las imágenes y recortes PNG extraídos se mantienen guardados.',
    };
  }

  const allDets = await idbGetAll<StoredDetection>('detections');
  const affected = new Set<string>();
  for (const d of allDets) {
    if (d.media_id === mediaId) {
      if (d.person_id) affected.add(d.person_id);
      await idbDelete('detections', d.id);
    }
  }
  await idbDelete('media', mediaId);
  for (const pid of affected) {
    await recomputeBrowserPersonStats(pid);
  }
  return { deleted: true, message: 'Archivo y detecciones eliminados completamente.' };
}

// ---------- Detections ----------
export async function saveDetection(det: StoredDetection): Promise<void> {
  await idbPut('detections', det);
}

export async function getAllStoredDetections(): Promise<StoredDetection[]> {
  return idbGetAll<StoredDetection>('detections');
}

export async function getStoredDetection(detId: string): Promise<StoredDetection | null> {
  return (await idbGet<StoredDetection>('detections', detId)) || null;
}

export async function deleteStoredDetection(detId: string): Promise<void> {
  const det = await getStoredDetection(detId);
  await idbDelete('detections', detId);
  if (det?.person_id) {
    await recomputeBrowserPersonStats(det.person_id);
  }
}

// ---------- Persons ----------
export async function savePerson(person: StoredPerson): Promise<void> {
  await idbPut('persons', person);
}

export async function getAllStoredPersons(): Promise<StoredPerson[]> {
  return idbGetAll<StoredPerson>('persons');
}

export async function getStoredPerson(personId: string): Promise<StoredPerson | null> {
  return (await idbGet<StoredPerson>('persons', personId)) || null;
}

export async function recomputeBrowserPersonStats(personId: string): Promise<void> {
  const person = await getStoredPerson(personId);
  if (!person) return;
  const allDets = (await getAllStoredDetections())
    .filter((d) => d.person_id === personId)
    .sort((a, b) => b.overall_score - a.overall_score);

  if (allDets.length === 0) {
    // Preserve person's saved centroid & reference_embeddings even with 0 photos!
    await savePerson({
      ...person,
      cover_detection_id: null,
      best_portrait_id: null,
      best_body_id: null,
      updated_at: new Date().toISOString(),
    });
    return;
  }

  // Weighted average centroid + top 6 reference embeddings preserved on the person record
  const top = allDets
    .slice(0, 20)
    .filter((d) => Array.isArray(d.embedding) && d.embedding.length > 0);
  let centroid = person.centroid;
  let refEmbeddings = person.reference_embeddings || [];
  if (top.length > 0) {
    const dim = top[0].embedding!.length;
    const sum = new Array<number>(dim).fill(0);
    let wTotal = 0;
    for (const d of top) {
      const w = Math.max(0.1, d.portrait_score / 100);
      wTotal += w;
      for (let i = 0; i < dim; i++) {
        sum[i] += d.embedding![i] * w;
      }
    }
    if (wTotal > 0) {
      centroid = sum.map((v) => v / wTotal);
    }
    refEmbeddings = top
      .sort((a, b) => b.portrait_score - a.portrait_score)
      .slice(0, 6)
      .map((d) => [...d.embedding!]);
  }

  const kept = allDets.filter((d) => d.status === 'kept');
  const pool = kept.length > 0 ? kept : allDets;
  const bestPortrait = pool.reduce((best, cur) =>
    cur.portrait_score > best.portrait_score ? cur : best
  ).id;

  const bodyPool = pool.filter((d) => d.has_cutout || d.body_score > 0);
  const bestBody =
    bodyPool.length > 0
      ? bodyPool.reduce((best, cur) => (cur.body_score > best.body_score ? cur : best)).id
      : null;

  const validIds = new Set(allDets.map((d) => d.id));
  const coverId =
    person.cover_detection_id && validIds.has(person.cover_detection_id)
      ? person.cover_detection_id
      : bestPortrait;

  await savePerson({
    ...person,
    centroid,
    reference_embeddings: refEmbeddings,
    best_portrait_id: bestPortrait,
    best_body_id: bestBody,
    cover_detection_id: coverId,
    updated_at: new Date().toISOString(),
  });
}

export async function clearBrowserPersonPhotos(
  personId: string
): Promise<{ cleared_detections: number; person_preserved: boolean }> {
  const person = await getStoredPerson(personId);
  if (!person) throw new Error('Persona no encontrada');

  // Snapshot centroid & reference embeddings BEFORE deleting any photos so recognition stays 100% intact
  await recomputeBrowserPersonStats(personId);
  const refreshedPerson = (await getStoredPerson(personId)) || person;

  const allDets = await getAllStoredDetections();
  const mine = allDets.filter((d) => d.person_id === personId);
  for (const d of mine) {
    for (const key of [d.portrait_path, d.cutout_path, d.crop_path, d.thumb_path]) {
      if (key) await deleteBlob(key);
    }
    await idbDelete('detections', d.id);
  }

  await savePerson({
    ...refreshedPerson,
    cover_detection_id: null,
    best_portrait_id: null,
    best_body_id: null,
    updated_at: new Date().toISOString(),
  });

  return { cleared_detections: mine.length, person_preserved: true };
}

function stripEmbedding(d: StoredDetection, mediaMap: Map<string, StoredMedia>): Detection {
  const { embedding: _e, ...rest } = d;
  const m = mediaMap.get(d.media_id);
  return {
    ...rest,
    media_filename: m?.original_filename || rest.media_filename,
    media_type: m?.media_type || rest.media_type,
    source_deleted: m ? Boolean(m.source_deleted) : Boolean(rest.source_deleted),
  };
}

function hydratePersonSummary(
  p: StoredPerson,
  allDets: StoredDetection[],
  mediaMap: Map<string, StoredMedia>
): PersonSummary {
  const mine = allDets.filter((d) => d.person_id === p.id);
  const kept = mine.filter((d) => d.status === 'kept');
  const discarded = mine.filter((d) => d.status === 'discarded');
  const cutouts = kept.filter((d) => d.has_cutout);
  const mediaSet = new Set(mine.map((d) => d.media_id));
  const detMap = new Map(mine.map((d) => [d.id, d]));

  const pick = (id: string | null) =>
    id && detMap.has(id) ? stripEmbedding(detMap.get(id)!, mediaMap) : null;

  return {
    id: p.id,
    name: p.name,
    notes: p.notes,
    cover_detection_id: p.cover_detection_id,
    best_portrait_id: p.best_portrait_id,
    best_body_id: p.best_body_id,
    is_ignored: p.is_ignored,
    created_at: p.created_at,
    updated_at: p.updated_at,
    total_detections: mine.length,
    kept_count: kept.length,
    discarded_count: discarded.length,
    cutout_count: cutouts.length,
    max_portrait_score:
      mine.length > 0 ? Math.round(Math.max(...mine.map((d) => d.portrait_score)) * 10) / 10 : 0,
    max_body_score:
      mine.length > 0 ? Math.round(Math.max(...mine.map((d) => d.body_score)) * 10) / 10 : 0,
    media_count: mediaSet.size,
    cover_detection: pick(p.cover_detection_id),
    best_portrait: pick(p.best_portrait_id),
    best_body: pick(p.best_body_id),
  };
}

export async function listBrowserPersons(includeIgnored = false): Promise<PersonSummary[]> {
  await ensureBlobsHydrated();
  const [persons, allDets, allMedia] = await Promise.all([
    getAllStoredPersons(),
    getAllStoredDetections(),
    idbGetAll<StoredMedia>('media'),
  ]);
  const mediaMap = new Map(allMedia.map((m) => [m.id, m]));
  return persons
    .filter((p) => (includeIgnored ? true : !p.is_ignored))
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .map((p) => hydratePersonSummary(p, allDets, mediaMap));
}

export async function getBrowserPersonDetail(personId: string): Promise<PersonDetail> {
  await ensureBlobsHydrated();
  const [person, allDets, allMedia] = await Promise.all([
    getStoredPerson(personId),
    getAllStoredDetections(),
    idbGetAll<StoredMedia>('media'),
  ]);
  if (!person) throw new Error('Persona no encontrada');
  const mediaMap = new Map(allMedia.map((m) => [m.id, m]));
  const summary = hydratePersonSummary(person, allDets, mediaMap);

  const mine = allDets
    .filter((d) => d.person_id === personId)
    .sort((a, b) => b.overall_score - a.overall_score)
    .map((d) => stripEmbedding(d, mediaMap));

  const kept = mine.filter((d) => d.status === 'kept');
  const discarded = mine.filter((d) => d.status === 'discarded');

  const byMedia = new Map<string, MediaAppearance>();
  for (const d of mine) {
    const m = mediaMap.get(d.media_id);
    if (!byMedia.has(d.media_id)) {
      byMedia.set(d.media_id, {
        media_id: d.media_id,
        filename: m?.original_filename || d.media_filename || 'archivo',
        media_type: m?.media_type || d.media_type || 'image',
        source_deleted: Boolean(m?.source_deleted),
        duration_sec: m?.duration_sec ?? null,
        appearances_count: 0,
        best_score: d.overall_score,
        timestamps: [],
      });
    }
    const entry = byMedia.get(d.media_id)!;
    entry.best_score = Math.max(entry.best_score || 0, d.overall_score);
    entry.timestamps.push({
      detection_id: d.id,
      timestamp_sec: d.timestamp_sec,
      frame_idx: d.frame_idx,
      portrait_score: d.portrait_score,
      body_score: d.body_score,
      status: d.status,
      thumb_path: d.thumb_path,
    });
  }

  const appearances = Array.from(byMedia.values())
    .map((item) => ({
      ...item,
      appearances_count: item.timestamps.length,
      timestamps: item.timestamps.sort((a, b) => a.timestamp_sec - b.timestamp_sec),
    }))
    .sort((a, b) => (b.best_score || 0) - (a.best_score || 0));

  return {
    ...summary,
    kept_detections: kept,
    discarded_detections: discarded,
    media_appearances: appearances,
  };
}

export async function updateBrowserPerson(
  personId: string,
  payload: { name?: string; notes?: string; cover_detection_id?: string; is_ignored?: boolean }
): Promise<PersonSummary> {
  const existing = await getStoredPerson(personId);
  if (!existing) throw new Error('Persona no encontrada');
  await savePerson({
    ...existing,
    name: payload.name !== undefined ? payload.name.trim() : existing.name,
    notes: payload.notes !== undefined ? payload.notes : existing.notes,
    cover_detection_id:
      payload.cover_detection_id !== undefined
        ? payload.cover_detection_id
        : existing.cover_detection_id,
    is_ignored:
      payload.is_ignored !== undefined ? (payload.is_ignored ? 1 : 0) : existing.is_ignored,
    updated_at: new Date().toISOString(),
  });
  const detail = await getBrowserPersonDetail(personId);
  return detail;
}

export async function mergeBrowserPersons(
  targetId: string,
  sourceIds: string[]
): Promise<PersonSummary> {
  const cleanSources = new Set(sourceIds.filter((s) => s !== targetId));
  const allDets = await getAllStoredDetections();
  for (const d of allDets) {
    if (d.person_id && cleanSources.has(d.person_id)) {
      await saveDetection({ ...d, person_id: targetId });
    }
  }
  for (const sid of cleanSources) {
    await idbDelete('persons', sid);
  }
  await recomputeBrowserPersonStats(targetId);
  return getBrowserPersonDetail(targetId);
}

export async function deleteBrowserPerson(personId: string): Promise<{ deleted: boolean }> {
  const allDets = await getAllStoredDetections();
  for (const d of allDets) {
    if (d.person_id === personId) {
      await idbDelete('detections', d.id);
    }
  }
  await idbDelete('persons', personId);
  return { deleted: true };
}
