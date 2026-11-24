import JSZip from 'jszip';
import type {
  AppSettings,
  Detection,
  JobInfo,
  JobSummaryMediaResult,
  MediaAppearance,
  SearchResult,
  SystemStats,
} from '../types';
import {
  type StoredDetection,
  type StoredMedia,
  type StoredPerson,
  deleteBrowserPerson,
  ensureBlobsHydrated,
  getAllStoredDetections,
  getAllStoredPersons,
  getBrowserPersonDetail,
  getBrowserSettings,
  getMediaRecord,
  getStoredDetection,
  listBrowserMedia,
  listBrowserPersons,
  loadBlob,
  purgeExpiredBrowserSources,
  recomputeBrowserPersonStats,
  saveBlob,
  saveDetection,
  saveMedia,
  savePerson,
} from './db';
import {
  type BrowserQualityMetrics,
  analyzeFrameBrowser,
  buildBrowserSceneSummary,
  descriptorSimilarity,
  detectFacesOnlyBrowser,
  euclideanDistance,
  extractAllAssetBlobsBrowser,
  extractThumbBlobBrowser,
  scoreDetectionBrowser,
} from './ml';

const jobsStore = new Map<string, JobInfo>();

function uid(): string {
  return (
    Math.random().toString(36).slice(2, 8) + Date.now().toString(36).slice(-6)
  ).slice(0, 12);
}

function classifyFile(file: File): 'image' | 'video' | null {
  if (
    file.type.startsWith('image/') ||
    /\.(jpg|jpeg|png|webp|bmp|gif|avif|heic|heif)$/i.test(file.name)
  ) {
    return 'image';
  }
  if (
    file.type.startsWith('video/') ||
    /\.(mp4|mov|avi|mkv|webm|m4v|3gp|3gpp|mpeg|mpg|ogv|ts)$/i.test(file.name)
  ) {
    return 'video';
  }
  return null;
}

interface PersonMatchEntry {
  name: string;
  vec: number[];
  exemplars: number[][];
  count: number;
}

function loadImageToCanvas(blob: Blob, maxDim = 1600): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(1, maxDim / Math.max(1, img.naturalWidth, img.naturalHeight));
      const w = Math.max(1, Math.round(img.naturalWidth * scale));
      const h = Math.max(1, Math.round(img.naturalHeight * scale));
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      canvas.getContext('2d')!.drawImage(img, 0, 0, w, h);
      resolve(canvas);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('No se pudo decodificar la imagen'));
    };
    img.src = url;
  });
}

async function makeMediaThumb(canvas: HTMLCanvasElement, mediaId: string): Promise<string> {
  const scale = Math.min(1, 480 / Math.max(1, canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * scale));
  const h = Math.max(1, Math.round(canvas.height * scale));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  c.getContext('2d')!.drawImage(canvas, 0, 0, w, h);
  const blob = await new Promise<Blob>((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error('thumb error'))), 'image/jpeg', 0.82)
  );
  const key = `media_${mediaId}.jpg`;
  await saveBlob(key, blob);
  return key;
}

async function matchOrCreateBrowserPerson(
  embedding: number[],
  faceDetScore: number,
  metrics: BrowserQualityMetrics,
  threshold: number,
  centroids: Map<string, PersonMatchEntry>,
  excludePersonIds: Set<string> = new Set()
): Promise<{ personId: string; personName: string; isNew: boolean }> {
  // Calibrated dlib 128-D Euclidean threshold (0.54 at default threshold=0.45):
  // Empirically, same-person different-photo distances span 0.35–0.54 while different people sit >= 0.60.
  // Combined with per-frame Cannot-Link exclusion + runner-up margin guard, this unites real matches
  // across smiles/angles/lighting without merging different friends in group photos.
  const baseMaxDist = Math.max(0.42, Math.min(0.62, 0.54 - (threshold - 0.45) * 0.45));
  const isVeryLowQuality =
    faceDetScore < 0.45 || metrics.portrait_score < 32 || metrics.frontality_score < 30;
  const isCleanAnchor =
    faceDetScore >= 0.55 && metrics.portrait_score >= 44 && metrics.frontality_score >= 40;
  const effectiveMaxDist = isVeryLowQuality ? Math.max(0.40, baseMaxDist - 0.025) : baseMaxDist;

  let bestId: string | null = null;
  let bestName = '';
  let bestCombined = Infinity;
  let bestExMin = Infinity;
  let secondBestCombined = Infinity;

  for (const [pid, info] of centroids.entries()) {
    // Cannot-Link: two faces appearing in the SAME photo/frame cannot be the same person
    if (excludePersonIds.has(pid)) continue;

    const centroidDist = euclideanDistance(embedding, info.vec);
    const exDists = info.exemplars
      .map((ex) => euclideanDistance(embedding, ex))
      .sort((a, b) => a - b);
    const bestEx = exDists.length > 0 ? exDists[0] : centroidDist;
    const topK = exDists.slice(0, Math.min(3, exDists.length));
    const avgTopDist =
      topK.length > 0 ? topK.reduce((s, d) => s + d, 0) / topK.length : centroidDist;
    const combinedDist = 0.45 * centroidDist + 0.35 * avgTopDist + 0.20 * bestEx;

    const directMatch =
      centroidDist <= effectiveMaxDist + 0.03 && avgTopDist <= effectiveMaxDist + 0.02;
    const strongExemplarMatch =
      bestEx <= effectiveMaxDist - 0.05 && centroidDist <= effectiveMaxDist + 0.06;

    if (directMatch || strongExemplarMatch) {
      if (combinedDist < bestCombined) {
        secondBestCombined = bestCombined;
        bestCombined = combinedDist;
        bestExMin = bestEx;
        bestId = pid;
        bestName = info.name;
      } else if (combinedDist < secondBestCombined) {
        secondBestCombined = combinedDist;
      }
    }
  }

  // Ambiguity guard in the borderline zone: if two different people are nearly tied, require a clear exemplar match
  const isBorderline = bestCombined > effectiveMaxDist - 0.03;
  const isAmbiguousTie =
    isBorderline &&
    secondBestCombined < Infinity &&
    secondBestCombined - bestCombined < 0.03 &&
    bestExMin > effectiveMaxDist - 0.04;

  if (bestId && bestCombined <= effectiveMaxDist + 0.015 && !isAmbiguousTie) {
    // Update running centroid & diverse exemplars for clean shots so future angles match even better
    if (isCleanAnchor) {
      const cur = centroids.get(bestId)!;
      const nextCount = cur.count + 1;
      const nextVec = cur.vec.map((v, i) => (v * cur.count + embedding[i]) / nextCount);
      const nextExemplars =
        cur.exemplars.length < 10 ? [...cur.exemplars, [...embedding]] : cur.exemplars;
      centroids.set(bestId, {
        name: cur.name,
        vec: nextVec,
        exemplars: nextExemplars,
        count: nextCount,
      });
    }
    return { personId: bestId, personName: bestName, isNew: false };
  }

  const allPersons = await getAllStoredPersons();
  const newName = `Persona ${allPersons.length + 1}`;
  const now = new Date().toISOString();
  const person: StoredPerson = {
    id: uid(),
    name: newName,
    notes: '',
    centroid: [...embedding],
    cover_detection_id: null,
    best_portrait_id: null,
    best_body_id: null,
    is_ignored: 0,
    created_at: now,
    updated_at: now,
  };
  await savePerson(person);
  centroids.set(person.id, {
    name: newName,
    vec: [...embedding],
    exemplars: [[...embedding]],
    count: 1,
  });
  return { personId: person.id, personName: newName, isNew: true };
}

async function evaluateKeepOrDiscardBrowser(
  personId: string,
  personName: string,
  mediaId: string,
  timestampSec: number,
  metrics: BrowserQualityMetrics,
  settings: AppSettings
): Promise<{ status: 'kept' | 'discarded'; keepCategory: string | null; discardReason: string | null }> {
  const minQuality = Number(settings.min_quality_score || '32');
  const maxPortraits = Number(settings.max_portraits_per_person || '6');
  const maxBodies = Number(settings.max_fullbody_per_person || '4');
  const minGap = Number(settings.min_video_gap_seconds || '1.5');

  const existingKept = (await getAllStoredDetections()).filter(
    (d) => d.person_id === personId && d.status === 'kept'
  );

  // NEVER silently discard a person's very first / only capture!
  // Every detected person always keeps at least their best capture so the user can see and inspect them.
  if (existingKept.length === 0) {
    return { status: 'kept', keepCategory: 'portrait', discardReason: null };
  }

  if (metrics.portrait_score < minQuality && metrics.body_score < minQuality + 8) {
    const reasons: string[] = [];
    if (metrics.sharpness_score < 35) reasons.push('borrosa');
    if (metrics.frontality_score < 40) reasons.push('muy de perfil');
    const detail = reasons.length > 0 ? ` (${reasons.join(', ')})` : '';
    return {
      status: 'discarded',
      keepCategory: null,
      discardReason: `Calidad insuficiente (${metrics.overall_score.toFixed(0)}/100)${detail}`,
    };
  }

  if (minGap > 0) {
    for (const ex of existingKept) {
      if (ex.media_id === mediaId && Math.abs(ex.timestamp_sec - timestampSec) < minGap) {
        if (
          metrics.portrait_score <= ex.portrait_score + 4 &&
          metrics.body_score <= ex.body_score + 4
        ) {
          return {
            status: 'discarded',
            keepCategory: null,
            discardReason: `Frame muy cercano (${timestampSec.toFixed(1)}s) a una captura mejor de ${personName}`,
          };
        }
      }
    }
  }

  const portraitSorted = [...existingKept].sort((a, b) => b.portrait_score - a.portrait_score);
  const qualifiesPortrait =
    portraitSorted.length < maxPortraits ||
    metrics.portrait_score > portraitSorted[maxPortraits - 1].portrait_score;

  const bodySorted = existingKept
    .filter((d) => d.has_cutout || d.body_score > 0)
    .sort((a, b) => b.body_score - a.body_score);
  const hasUsableBody = metrics.body_score >= minQuality;
  const qualifiesBody =
    hasUsableBody &&
    (bodySorted.length < maxBodies ||
      metrics.body_score > bodySorted[maxBodies - 1].body_score);

  if (qualifiesPortrait && qualifiesBody) {
    return { status: 'kept', keepCategory: 'both', discardReason: null };
  }
  if (qualifiesPortrait) {
    return { status: 'kept', keepCategory: 'portrait', discardReason: null };
  }
  if (qualifiesBody) {
    return { status: 'kept', keepCategory: 'full_body', discardReason: null };
  }

  const bestP = portraitSorted[0]?.portrait_score ?? 0;
  const bestB = bodySorted[0]?.body_score ?? 0;
  return {
    status: 'discarded',
    keepCategory: null,
    discardReason: `Descartada: ${personName} ya tiene capturas mejores (retrato ${metrics.portrait_score.toFixed(0)} vs ${bestP.toFixed(0)}, cuerpo ${metrics.body_score.toFixed(0)} vs ${bestB.toFixed(0)})`,
  };
}

async function pruneExcessKeptBrowser(
  personId: string,
  personName: string,
  settings: AppSettings
): Promise<void> {
  const maxPortraits = Number(settings.max_portraits_per_person || '6');
  const maxBodies = Number(settings.max_fullbody_per_person || '4');
  const kept = (await getAllStoredDetections()).filter(
    (d) => d.person_id === personId && d.status === 'kept'
  );
  if (kept.length <= Math.max(maxPortraits, maxBodies)) return;

  const topP = new Set(
    [...kept]
      .sort((a, b) => b.portrait_score - a.portrait_score)
      .slice(0, maxPortraits)
      .map((d) => d.id)
  );
  const topB = new Set(
    kept
      .filter((d) => d.has_cutout || d.body_score > 0)
      .sort((a, b) => b.body_score - a.body_score)
      .slice(0, maxBodies)
      .map((d) => d.id)
  );
  const bestP = Math.max(...kept.map((d) => d.portrait_score), 0);

  for (const d of kept) {
    if (!topP.has(d.id) && !topB.has(d.id)) {
      await saveDetection({
        ...d,
        status: 'discarded',
        discard_reason: `Sustituida por capturas más claras de ${personName} (retrato ${d.portrait_score.toFixed(0)} vs ${bestP.toFixed(0)})`,
      });
    }
  }
}

async function processFrameCanvas(
  canvas: HTMLCanvasElement,
  media: StoredMedia,
  timestampSec: number,
  frameIdx: number,
  settings: AppSettings,
  centroids: Map<string, PersonMatchEntry>,
  affectedPersons: Set<string>,
  batchPersonTracker: Map<string, { isNew: boolean; count: number }>,
  onStatus?: (msg: string) => void
): Promise<{ total: number; kept: number; discarded: number; newPersons: number }> {
  const analysis = await analyzeFrameBrowser(canvas, onStatus);
  if (analysis.faces.length === 0) {
    return { total: 0, kept: 0, discarded: 0, newPersons: 0 };
  }

  const usedBodies = new Set<number>();
  const sortedFaces = [...analysis.faces].sort(
    (a, b) =>
      (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) -
      (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
  );

  const threshold = Number(settings.similarity_threshold || '0.45');
  const frameAssignedPersons = new Set<string>();
  let keptCount = 0;
  let discardedCount = 0;
  let newPersons = 0;
  let sharedFrameKey: string | null = null;

  for (const face of sortedFaces) {
    const metrics = scoreDetectionBrowser(
      canvas,
      face,
      analysis.bodies,
      Boolean(analysis.personMask),
      usedBodies
    );
    if (metrics.matchedBodyIdx !== null) usedBodies.add(metrics.matchedBodyIdx);

    const { personId, personName, isNew } = await matchOrCreateBrowserPerson(
      face.embedding,
      face.detScore,
      metrics,
      threshold,
      centroids,
      frameAssignedPersons
    );
    frameAssignedPersons.add(personId);
    affectedPersons.add(personId);
    if (isNew) newPersons++;
    const prevTrack = batchPersonTracker.get(personId);
    batchPersonTracker.set(personId, {
      isNew: prevTrack ? prevTrack.isNew : isNew,
      count: (prevTrack?.count ?? 0) + 1,
    });

    const decision = await evaluateKeepOrDiscardBrowser(
      personId,
      personName,
      media.id,
      timestampSec,
      metrics,
      settings
    );

    const detId = uid();
    const { summary, tags } = buildBrowserSceneSummary(
      face,
      Math.max(analysis.faces.length, analysis.bodies.length),
      metrics.is_full_body,
      analysis.objects
    );

    if (decision.status === 'kept') {
      const blobs = await extractAllAssetBlobsBrowser(
        canvas,
        face.bbox,
        metrics.matchedBody,
        analysis.bodies,
        analysis.personMask,
        analysis.maskWidth,
        analysis.maskHeight
      );

      if (!sharedFrameKey) {
        sharedFrameKey = `${detId}_frame.jpg`;
        await saveBlob(sharedFrameKey, blobs.frameBlob);
      }
      const portraitKey = `${detId}_portrait.jpg`;
      const cropKey = `${detId}_crop.jpg`;
      const thumbKey = `${detId}_thumb.jpg`;
      await Promise.all([
        saveBlob(portraitKey, blobs.portraitBlob),
        saveBlob(cropKey, blobs.cropBlob),
        saveBlob(thumbKey, blobs.thumbBlob),
      ]);

      let cutoutKey: string | null = null;
      if (blobs.cutoutBlob) {
        cutoutKey = `${detId}_cutout.png`;
        await saveBlob(cutoutKey, blobs.cutoutBlob);
      }

      const det: StoredDetection = {
        id: detId,
        media_id: media.id,
        person_id: personId,
        media_filename: media.original_filename,
        media_type: media.media_type,
        source_deleted: false,
        timestamp_sec: Math.round(timestampSec * 100) / 100,
        frame_idx: frameIdx,
        face_bbox: face.bbox.map((v) => Math.round(v * 10) / 10) as [number, number, number, number],
        body_bbox: metrics.matchedBody
          ? (metrics.matchedBody.bbox.map((v) => Math.round(v * 10) / 10) as [
              number,
              number,
              number,
              number,
            ])
          : null,
        det_score: Math.round(face.detScore * 1000) / 1000,
        portrait_score: metrics.portrait_score,
        body_score: metrics.body_score,
        overall_score: metrics.overall_score,
        sharpness_score: metrics.sharpness_score,
        frontality_score: metrics.frontality_score,
        separation_score: metrics.separation_score,
        has_cutout: Boolean(cutoutKey),
        is_full_body: metrics.is_full_body,
        frame_path: sharedFrameKey,
        portrait_path: portraitKey,
        cutout_path: cutoutKey,
        crop_path: cropKey,
        thumb_path: thumbKey,
        status: 'kept',
        keep_category: decision.keepCategory,
        discard_reason: null,
        scene_summary: summary,
        scene_tags: tags,
        scene_backend: 'browser-ai',
        created_at: new Date().toISOString(),
        embedding: face.embedding,
      };
      await saveDetection(det);
      await pruneExcessKeptBrowser(personId, personName, settings);
      keptCount++;
    } else {
      const thumbBlob = await extractThumbBlobBrowser(canvas, face.bbox);
      const thumbKey = `${detId}_thumb.jpg`;
      await saveBlob(thumbKey, thumbBlob);

      const det: StoredDetection = {
        id: detId,
        media_id: media.id,
        person_id: personId,
        media_filename: media.original_filename,
        media_type: media.media_type,
        source_deleted: false,
        timestamp_sec: Math.round(timestampSec * 100) / 100,
        frame_idx: frameIdx,
        face_bbox: face.bbox.map((v) => Math.round(v * 10) / 10) as [number, number, number, number],
        body_bbox: metrics.matchedBody
          ? (metrics.matchedBody.bbox.map((v) => Math.round(v * 10) / 10) as [
              number,
              number,
              number,
              number,
            ])
          : null,
        det_score: Math.round(face.detScore * 1000) / 1000,
        portrait_score: metrics.portrait_score,
        body_score: metrics.body_score,
        overall_score: metrics.overall_score,
        sharpness_score: metrics.sharpness_score,
        frontality_score: metrics.frontality_score,
        separation_score: metrics.separation_score,
        has_cutout: false,
        is_full_body: metrics.is_full_body,
        frame_path: null,
        portrait_path: null,
        cutout_path: null,
        crop_path: null,
        thumb_path: thumbKey,
        status: 'discarded',
        keep_category: null,
        discard_reason: decision.discardReason,
        scene_summary: summary,
        scene_tags: tags,
        scene_backend: 'browser-ai',
        created_at: new Date().toISOString(),
        embedding: face.embedding,
      };
      await saveDetection(det);
      discardedCount++;
    }
  }

  return { total: sortedFaces.length, kept: keptCount, discarded: discardedCount, newPersons };
}

function seekVideoTo(video: HTMLVideoElement, sec: number, timeoutMs = 4500): Promise<boolean> {
  return new Promise((resolve) => {
    const target = Math.min(Math.max(0, sec), Math.max(0, (video.duration || 0) - 0.05));
    if (Math.abs(video.currentTime - target) < 0.02 && video.readyState >= 2) {
      resolve(true);
      return;
    }

    let settled = false;
    const finish = (ok: boolean) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
      // Let mobile hardware video decoder paint frame before canvas drawImage
      window.requestAnimationFrame(() => resolve(ok));
    };
    const onSeeked = () => finish(true);
    const onError = () => finish(false);
    const timer = window.setTimeout(() => finish(video.readyState >= 2), timeoutMs);

    video.addEventListener('seeked', onSeeked);
    video.addEventListener('error', onError);
    try {
      video.currentTime = target;
    } catch {
      finish(false);
    }
  });
}

async function processVideoInBrowser(
  blob: Blob,
  media: StoredMedia,
  sampleFps: number,
  settings: AppSettings,
  centroids: Map<string, PersonMatchEntry>,
  affectedPersons: Set<string>,
  batchPersonTracker: Map<string, { isNew: boolean; count: number }>,
  onProgress: (frac: number, step: string) => void
): Promise<{
  width: number;
  height: number;
  durationSec: number;
  thumbKey: string | null;
  framesAnalyzed: number;
  total: number;
  kept: number;
  discarded: number;
  newPersons: number;
}> {
  const url = URL.createObjectURL(blob);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.setAttribute('playsinline', 'true');
  video.setAttribute('webkit-playsinline', 'true');
  video.preload = 'auto';
  // Mounting invisibly in DOM is required on iOS Safari so hardware decoding renders video frames
  video.style.position = 'fixed';
  video.style.left = '-9999px';
  video.style.top = '0';
  video.style.width = '2px';
  video.style.height = '2px';
  video.style.opacity = '0.001';
  video.style.pointerEvents = 'none';
  document.body.appendChild(video);

  let thumbKey: string | null = null;
  let total = 0;
  let kept = 0;
  let discarded = 0;
  let newPersons = 0;
  let analyzedCount = 0;
  let duration = 1;
  let w = 640;
  let h = 480;

  try {
    video.src = url;
    video.load();

    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(
        () => reject(new Error('El navegador tardó demasiado en abrir este vídeo (prueba MP4/H.264)')),
        15000
      );
      const done = () => {
        window.clearTimeout(timer);
        video.removeEventListener('loadeddata', done);
        video.removeEventListener('loadedmetadata', onMeta);
        video.removeEventListener('error', onErr);
        resolve();
      };
      const onMeta = () => {
        if (video.readyState >= 2) done();
      };
      const onErr = () => {
        window.clearTimeout(timer);
        reject(
          new Error(
            'Tu navegador no pudo decodificar este formato de vídeo. En móvil prueba con MP4/MOV estándar o usa el servidor Python.'
          )
        );
      };
      video.addEventListener('loadeddata', done);
      video.addEventListener('loadedmetadata', onMeta);
      video.addEventListener('error', onErr);
    });

    // Warm up iOS Safari hardware video decoder briefly (muted)
    try {
      const playPromise = video.play();
      if (playPromise) {
        await playPromise.catch(() => undefined);
      }
      video.pause();
    } catch {
      // ignore autoplay restrictions
    }

    duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
    const rawW = video.videoWidth || 640;
    const rawH = video.videoHeight || 480;
    // Cap video analysis canvas at 1200px so smaller/further-back faces remain detectable
    const scale = Math.min(1, 1200 / Math.max(1, rawW, rawH));
    w = Math.max(1, Math.round(rawW * scale));
    h = Math.max(1, Math.round(rawH * scale));

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;

    // Compute sampling timestamps (cap at 180 frames so long videos stay snappy on mobile)
    let stepSec = 1 / Math.max(0.2, sampleFps);
    const maxFrames = 180;
    if (duration / stepSec > maxFrames) {
      stepSec = duration / maxFrames;
    }
    const timestamps: number[] = [];
    for (let t = 0.05; t < duration; t += stepSec) {
      timestamps.push(t);
    }
    if (timestamps.length === 0) timestamps.push(0);

    for (let i = 0; i < timestamps.length; i++) {
      const ts = timestamps[i];
      const ok = await seekVideoTo(video, ts);
      if (!ok) continue;

      ctx.drawImage(video, 0, 0, w, h);
      analyzedCount++;
      if (!thumbKey) {
        thumbKey = await makeMediaThumb(canvas, media.id);
      }

      const res = await processFrameCanvas(
        canvas,
        media,
        ts,
        i,
        settings,
        centroids,
        affectedPersons,
        batchPersonTracker,
        (s) => onProgress(i / timestamps.length, s)
      );
      total += res.total;
      kept += res.kept;
      discarded += res.discarded;
      newPersons += res.newPersons;

      onProgress(
        (i + 1) / timestamps.length,
        `${media.original_filename}: ${ts.toFixed(1)}s / ${duration.toFixed(1)}s (${kept} mejores)`
      );

      // Yield briefly to keep mobile UI smooth and prevent browser watchdog kills
      await new Promise((r) => window.setTimeout(r, 15));
    }

    return {
      width: rawW,
      height: rawH,
      durationSec: Math.round(duration * 100) / 100,
      thumbKey,
      framesAnalyzed: analyzedCount,
      total,
      kept,
      discarded,
      newPersons,
    };
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    video.remove();
    URL.revokeObjectURL(url);
  }
}

export async function uploadMediaBrowser(
  files: File[],
  sampleFpsOverride?: number,
  keepForever = false
): Promise<{
  job_id: string;
  media_ids: string[];
  uploaded_count: number;
  keep_forever: boolean;
  skipped: Array<{ filename: string; reason: string }>;
}> {
  await ensureBlobsHydrated();
  const settings = await getBrowserSettings();
  const retentionMins = Math.max(1, Number(settings.source_retention_minutes || '60'));

  const validPairs: Array<{ file: File; media: StoredMedia }> = [];
  const skipped: Array<{ filename: string; reason: string }> = [];

  for (const file of files) {
    const mediaType = classifyFile(file);
    if (!mediaType) {
      skipped.push({ filename: file.name, reason: 'Formato no soportado' });
      continue;
    }
    const mediaId = uid();
    const uploadKey = `upload_${mediaId}`;
    // Avoid blowing mobile IndexedDB storage quota on huge 100-400MB phone videos:
    // keep playable in current session and only persist raw upload blob in IDB if <= 45 MB
    const saved = await saveBlob(uploadKey, file, { maxPersistBytes: 45 * 1024 * 1024 });

    const now = new Date();
    const media: StoredMedia = {
      id: mediaId,
      original_filename: file.name,
      media_type: mediaType,
      file_path: uploadKey,
      upload_blob_key: saved.persisted ? uploadKey : null,
      thumb_path: null,
      file_size: file.size,
      width: 0,
      height: 0,
      duration_sec: null,
      fps: null,
      frames_analyzed: 0,
      detections_count: 0,
      kept_count: 0,
      discarded_count: 0,
      source_deleted: 0,
      keep_forever: keepForever ? 1 : 0,
      expires_at: keepForever
        ? null
        : new Date(now.getTime() + retentionMins * 60000).toISOString(),
      status: 'processing',
      verdict: 'pending',
      verdict_reason: null,
      created_at: now.toISOString(),
    };
    await saveMedia(media);
    validPairs.push({ file, media });
  }

  if (validPairs.length === 0) {
    throw new Error('No se subió ningún archivo de imagen o vídeo válido.');
  }

  const jobId = uid();
  const nowIso = new Date().toISOString();
  const job: JobInfo = {
    id: jobId,
    job_type: 'ingest',
    status: 'queued',
    progress: 0.02,
    total_items: validPairs.length,
    processed_items: 0,
    current_step: 'Iniciando análisis IA en el navegador...',
    error: null,
    summary: null,
    created_at: nowIso,
    updated_at: nowIso,
  };
  jobsStore.set(jobId, job);

  const sampleFps = sampleFpsOverride || Number(settings.video_sample_fps || '1.0');

  // Run asynchronously so UI shows live progress bar smoothly
  void (async () => {
    const updateJob = (patch: Partial<JobInfo>) => {
      const cur = jobsStore.get(jobId);
      if (!cur) return;
      jobsStore.set(jobId, { ...cur, ...patch, updated_at: new Date().toISOString() });
    };

    try {
      updateJob({ status: 'running', current_step: 'Cargando modelos IA en tu dispositivo...' });

      // Seed centroids AND top 6 saved exemplars per existing person for reliable automatic recognition
      const [existingPersons, existingDets] = await Promise.all([
        getAllStoredPersons(),
        getAllStoredDetections(),
      ]);
      const exemplarsByPerson = new Map<string, number[][]>();
      for (const d of [...existingDets].sort((a, b) => b.portrait_score - a.portrait_score)) {
        if (!d.person_id || !Array.isArray(d.embedding) || d.embedding.length === 0) continue;
        const arr = exemplarsByPerson.get(d.person_id) || [];
        if (arr.length < 6) {
          arr.push(d.embedding);
          exemplarsByPerson.set(d.person_id, arr);
        }
      }

      const centroids = new Map<string, PersonMatchEntry>();
      for (const p of existingPersons) {
        if (Array.isArray(p.centroid) && p.centroid.length > 0) {
          const detExemplars = exemplarsByPerson.get(p.id) || [];
          const savedRefs = Array.isArray(p.reference_embeddings) ? p.reference_embeddings : [];
          const exemplars =
            detExemplars.length > 0
              ? detExemplars
              : savedRefs.length > 0
                ? savedRefs
                : [[...p.centroid]];
          centroids.set(p.id, {
            name: p.name,
            vec: [...p.centroid],
            exemplars,
            count: Math.max(3, exemplars.length),
          });
        }
      }

      const affectedPersons = new Set<string>();
      const batchPersonTracker = new Map<string, { isNew: boolean; count: number }>();
      const mediaResults: JobSummaryMediaResult[] = [];
      let usefulMedia = 0;
      let discardedMedia = 0;
      let noPeopleMedia = 0;
      let totalFrames = 0;
      let totalDetections = 0;
      let keptDetections = 0;
      let discardedDetections = 0;
      let newPersonsTotal = 0;

      for (let idx = 0; idx < validPairs.length; idx++) {
        const { file, media } = validPairs[idx];
        const baseFrac = idx / validPairs.length;
        const span = 1 / validPairs.length;
        updateJob({
          progress: Math.max(0.05, baseFrac),
          processed_items: idx,
          current_step: `Analizando ${file.name} (${idx + 1}/${validPairs.length})...`,
        });

        let total = 0;
        let kept = 0;
        let discarded = 0;
        let framesAnalyzed = 0;
        let width = 0;
        let height = 0;
        let durationSec: number | null = null;
        let thumbKey: string | null = null;

        if (media.media_type === 'image') {
          const canvas = await loadImageToCanvas(file, 1600);
          width = canvas.width;
          height = canvas.height;
          thumbKey = await makeMediaThumb(canvas, media.id);
          const res = await processFrameCanvas(
            canvas,
            media,
            0,
            0,
            settings,
            centroids,
            affectedPersons,
            batchPersonTracker,
            (step) => updateJob({ current_step: step })
          );
          total = res.total;
          kept = res.kept;
          discarded = res.discarded;
          newPersonsTotal += res.newPersons;
          framesAnalyzed = 1;
        } else {
          const vres = await processVideoInBrowser(
            file,
            media,
            sampleFps,
            settings,
            centroids,
            affectedPersons,
            batchPersonTracker,
            (frac, step) => updateJob({ progress: baseFrac + span * frac, current_step: step })
          );
          width = vres.width;
          height = vres.height;
          durationSec = vres.durationSec;
          thumbKey = vres.thumbKey;
          framesAnalyzed = vres.framesAnalyzed;
          total = vres.total;
          kept = vres.kept;
          discarded = vres.discarded;
          newPersonsTotal += vres.newPersons;
        }

        let verdict: StoredMedia['verdict'] = 'no_people';
        let verdictReason = 'No se detectaron rostros claros en este archivo';
        if (total === 0) {
          noPeopleMedia++;
        } else if (kept > 0) {
          verdict = 'useful';
          verdictReason = `Útil: ${kept} captura(s) de alta calidad extraída(s)${
            discarded > 0 ? ` (${discarded} descartada(s))` : ''
          }`;
          usefulMedia++;
        } else {
          verdict = 'discarded';
          verdictReason = `Descartada: las ${discarded} aparición(es) ya tenían capturas de mayor calidad`;
          discardedMedia++;
        }

        await saveMedia({
          ...media,
          width,
          height,
          duration_sec: durationSec,
          thumb_path: thumbKey,
          frames_analyzed: framesAnalyzed,
          detections_count: total,
          kept_count: kept,
          discarded_count: discarded,
          status: 'done',
          verdict,
          verdict_reason: verdictReason,
        });

        totalFrames += framesAnalyzed;
        totalDetections += total;
        keptDetections += kept;
        discardedDetections += discarded;
        mediaResults.push({
          media_id: media.id,
          filename: media.original_filename,
          media_type: media.media_type,
          verdict,
          verdict_reason: verdictReason,
          detections: total,
          kept,
          discarded,
        });
      }

      updateJob({ current_step: 'Refinando agrupación de personas (alta precisión)...' });
      await reclusterAllBrowserPersons();

      // Build detected_persons list from the refined pure clusters so user sees clean separated people
      const [latestSummaries, allFinalDets] = await Promise.all([
        listBrowserPersons(),
        getAllStoredDetections(),
      ]);
      const batchMediaIds = new Set(validPairs.map((p) => p.media.id));
      const preexistingIds = new Set(existingPersons.map((p) => p.id));
      const countsByPerson = new Map<string, number>();
      for (const d of allFinalDets) {
        if (batchMediaIds.has(d.media_id) && d.person_id) {
          countsByPerson.set(d.person_id, (countsByPerson.get(d.person_id) || 0) + 1);
        }
      }
      const detectedPersons = latestSummaries
        .filter((p) => countsByPerson.has(p.id))
        .map((p) => ({
          id: p.id,
          name: p.name,
          is_new: !preexistingIds.has(p.id),
          thumb_path:
            p.cover_detection?.portrait_path ||
            p.cover_detection?.thumb_path ||
            p.best_portrait?.portrait_path ||
            p.best_portrait?.thumb_path ||
            null,
          captures_in_batch: countsByPerson.get(p.id) || 1,
        }));

      updateJob({
        status: 'done',
        progress: 1,
        processed_items: validPairs.length,
        current_step: 'Análisis completado',
        summary: {
          total_media: validPairs.length,
          useful_media: usefulMedia,
          discarded_media: discardedMedia,
          no_people_media: noPeopleMedia,
          total_frames_analyzed: totalFrames,
          total_detections: totalDetections,
          kept_detections: keptDetections,
          discarded_detections: discardedDetections,
          new_persons: newPersonsTotal,
          media_results: mediaResults,
          detected_persons: detectedPersons,
        },
      });
    } catch (err) {
      updateJob({
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
        current_step: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  })();

  return {
    job_id: jobId,
    media_ids: validPairs.map((p) => p.media.id),
    uploaded_count: validPairs.length,
    keep_forever: keepForever,
    skipped,
  };
}

export function getBrowserJob(jobId: string): JobInfo {
  const job = jobsStore.get(jobId);
  if (!job) throw new Error('Trabajo no encontrado');
  return job;
}

export function listBrowserJobs(): JobInfo[] {
  return Array.from(jobsStore.values()).sort((a, b) => b.created_at.localeCompare(a.created_at));
}

export async function getBrowserStats(): Promise<SystemStats> {
  const [persons, media] = await Promise.all([listBrowserPersons(), listBrowserMedia()]);
  const photos = media.filter((m) => m.media_type === 'image');
  const videos = media.filter((m) => m.media_type === 'video');
  const activeJobs = listBrowserJobs().filter(
    (j) => j.status === 'queued' || j.status === 'running'
  );
  return {
    persons_count: persons.length,
    photos_count: photos.length,
    videos_count: videos.length,
    kept_captures: persons.reduce((s, p) => s + p.kept_count, 0),
    cutouts_count: persons.reduce((s, p) => s + p.cutout_count, 0),
    discarded_captures: persons.reduce((s, p) => s + p.discarded_count, 0),
    active_jobs: activeJobs,
    device: {
      torch_device: 'Navegador (WebGL/WASM)',
      onnx_provider: '100% en tu dispositivo',
      onnx_providers: 'browser',
    },
  };
}

export async function updateBrowserDetectionStatus(
  detId: string,
  status: 'kept' | 'discarded',
  discardReason?: string
): Promise<Detection> {
  const det = await getStoredDetection(detId);
  if (!det) throw new Error('Detección no encontrada');

  let updated: StoredDetection = {
    ...det,
    status,
    keep_category: status === 'kept' ? 'manual' : null,
    discard_reason: status === 'kept' ? null : discardReason || 'Descartada manualmente',
  };

  // If rescuing and missing full assets, regenerate them from the source blob if still available
  if (status === 'kept' && (!det.portrait_path || !det.frame_path)) {
    const media = await getMediaRecord(det.media_id);
    if (media && !media.source_deleted && media.upload_blob_key && det.face_bbox) {
      const srcBlob = await loadBlob(media.upload_blob_key);
      if (srcBlob) {
        let canvas: HTMLCanvasElement | null = null;
        if (media.media_type === 'image') {
          canvas = await loadImageToCanvas(srcBlob, 1600);
        } else {
          const url = URL.createObjectURL(srcBlob);
          try {
            const video = document.createElement('video');
            video.muted = true;
            video.playsInline = true;
            video.src = url;
            await new Promise<void>((res, rej) => {
              video.onloadedmetadata = () => res();
              video.onerror = () => rej();
            });
            await seekVideoTo(video, det.timestamp_sec);
            canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height);
          } finally {
            URL.revokeObjectURL(url);
          }
        }
        if (canvas) {
          const analysis = await analyzeFrameBrowser(canvas);
          const blobs = await extractAllAssetBlobsBrowser(
            canvas,
            det.face_bbox,
            analysis.bodies[0] || null,
            analysis.bodies,
            analysis.personMask,
            analysis.maskWidth,
            analysis.maskHeight
          );
          const frameKey = `${detId}_frame.jpg`;
          const portraitKey = `${detId}_portrait.jpg`;
          const cropKey = `${detId}_crop.jpg`;
          const thumbKey = `${detId}_thumb.jpg`;
          await Promise.all([
            saveBlob(frameKey, blobs.frameBlob),
            saveBlob(portraitKey, blobs.portraitBlob),
            saveBlob(cropKey, blobs.cropBlob),
            saveBlob(thumbKey, blobs.thumbBlob),
          ]);
          let cutoutKey = det.cutout_path;
          if (blobs.cutoutBlob) {
            cutoutKey = `${detId}_cutout.png`;
            await saveBlob(cutoutKey, blobs.cutoutBlob);
          }
          updated = {
            ...updated,
            frame_path: frameKey,
            portrait_path: portraitKey,
            crop_path: cropKey,
            thumb_path: thumbKey,
            cutout_path: cutoutKey,
            has_cutout: Boolean(cutoutKey),
          };
        }
      }
    }
  }

  await saveDetection(updated);
  if (det.person_id) await recomputeBrowserPersonStats(det.person_id);
  const { embedding: _e, ...clean } = updated;
  return clean;
}

export async function searchByImageBrowser(
  file: File,
  faceIndex = 0,
  threshold?: number
): Promise<SearchResult> {
  await ensureBlobsHydrated();
  await purgeExpiredBrowserSources();

  const canvas = await loadImageToCanvas(file, 1280);
  const faces = await detectFacesOnlyBrowser(canvas);
  if (faces.length === 0) {
    return {
      found_faces: 0,
      selected_face_index: 0,
      query_faces: [],
      matched_persons: [],
      matched_media: [],
      message: 'No se detectó ningún rostro claro en la imagen de búsqueda.',
    };
  }

  const sorted = [...faces].sort(
    (a, b) =>
      (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]) -
      (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
  );
  const chosenIdx = Math.max(0, Math.min(sorted.length - 1, faceIndex));
  const queryEmb = sorted[chosenIdx].embedding;

  const settings = await getBrowserSettings();
  const baseThresh =
    threshold !== undefined ? threshold : Number(settings.similarity_threshold || '0.45');
  const simThreshold = Math.max(0.42, Math.min(0.75, baseThresh + 0.04));

  const [storedPersons, allSummaries, allDets, allMedia] = await Promise.all([
    getAllStoredPersons(),
    listBrowserPersons(),
    getAllStoredDetections(),
    listBrowserMedia(),
  ]);

  const summaryMap = new Map(allSummaries.map((p) => [p.id, p]));
  const mediaMap = new Map(allMedia.map((m) => [m.id, m]));
  const personNameMap = new Map(storedPersons.map((p) => [p.id, p.name]));

  const matchedPersons = storedPersons
    .filter((p) => Array.isArray(p.centroid) && p.centroid.length > 0)
    .map((p) => ({
      summary: summaryMap.get(p.id),
      sim: descriptorSimilarity(queryEmb, p.centroid!),
    }))
    .filter((x) => x.summary && x.sim >= simThreshold)
    .map((x) => ({ ...x.summary!, similarity: Math.round(x.sim * 1000) / 10 }))
    .sort((a, b) => (b.similarity || 0) - (a.similarity || 0));

  const mediaHits = new Map<string, MediaAppearance>();
  for (const d of allDets) {
    if (!Array.isArray(d.embedding) || d.embedding.length === 0) continue;
    const sim = descriptorSimilarity(queryEmb, d.embedding);
    if (sim < simThreshold) continue;
    const simPct = Math.round(sim * 1000) / 10;
    const m = mediaMap.get(d.media_id);

    if (!mediaHits.has(d.media_id)) {
      mediaHits.set(d.media_id, {
        media_id: d.media_id,
        filename: m?.original_filename || d.media_filename || 'archivo',
        media_type: m?.media_type || d.media_type || 'image',
        source_deleted: Boolean(m?.source_deleted),
        duration_sec: m?.duration_sec ?? null,
        best_similarity: simPct,
        appearances_count: 0,
        timestamps: [],
      });
    }
    const entry = mediaHits.get(d.media_id)!;
    entry.best_similarity = Math.max(entry.best_similarity || 0, simPct);
    entry.timestamps.push({
      detection_id: d.id,
      timestamp_sec: d.timestamp_sec,
      frame_idx: d.frame_idx,
      similarity: simPct,
      portrait_score: d.portrait_score,
      person_id: d.person_id,
      person_name: d.person_id ? personNameMap.get(d.person_id) || 'Persona' : null,
      thumb_path: d.thumb_path,
      portrait_path: d.portrait_path,
      status: d.status,
    });
  }

  const matchedMedia = Array.from(mediaHits.values())
    .map((m) => ({
      ...m,
      appearances_count: m.timestamps.length,
      timestamps: m.timestamps.sort((a, b) => a.timestamp_sec - b.timestamp_sec),
    }))
    .sort((a, b) => (b.best_similarity || 0) - (a.best_similarity || 0));

  return {
    found_faces: sorted.length,
    selected_face_index: chosenIdx,
    query_faces: sorted.map((f, i) => ({
      index: i,
      bbox: f.bbox.map((v) => Math.round(v)) as [number, number, number, number],
      det_score: Math.round(f.detScore * 1000) / 1000,
    })),
    matched_persons: matchedPersons,
    matched_media: matchedMedia,
    message:
      matchedMedia.length > 0 || matchedPersons.length > 0
        ? `Encontrado en ${matchedMedia.length} archivo(s) y ${matchedPersons.length} perfil(es).`
        : 'Esta persona no aparece en ninguna foto ni vídeo indexado todavía.',
  };
}

export async function exportZipBrowser(payload: {
  person_ids?: string[] | null;
  include_portraits?: boolean;
  include_cutouts?: boolean;
  include_crops?: boolean;
  include_originals?: boolean;
  only_kept?: boolean;
}): Promise<void> {
  const zip = new JSZip();
  const allPersons = await listBrowserPersons();
  const targetPersons = payload.person_ids?.length
    ? allPersons.filter((p) => payload.person_ids!.includes(p.id))
    : allPersons;

  const manifest: Array<{ name: string; folder: string; captures_exported: number }> = [];

  for (const p of targetPersons) {
    const detail = await getBrowserPersonDetail(p.id);
    const folder = p.name.replace(/[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ _-]+/g, '').trim().replace(/\s+/g, '_') || 'Persona';
    const dets =
      payload.only_kept !== false
        ? detail.kept_detections
        : [...detail.kept_detections, ...detail.discarded_detections];

    const capturesMeta: Array<Record<string, unknown>> = [];

    for (let i = 0; i < dets.length; i++) {
      const d = dets[i];
      const prefix = `${String(i + 1).padStart(2, '0')}_score${Math.round(d.overall_score)}_${d.id.slice(0, 6)}`;
      const filesObj: Record<string, string> = {};

      if (payload.include_portraits !== false && d.portrait_path) {
        const b = await loadBlob(d.portrait_path);
        if (b) {
          const arc = `${folder}/portraits/${prefix}_portrait.jpg`;
          zip.file(arc, b);
          filesObj.portrait = arc;
        }
      }
      if (payload.include_cutouts !== false && d.cutout_path) {
        const b = await loadBlob(d.cutout_path);
        if (b) {
          const arc = `${folder}/cutouts_png/${prefix}_cutout.png`;
          zip.file(arc, b);
          filesObj.cutout_png = arc;
        }
      }
      if (payload.include_crops !== false && d.crop_path) {
        const b = await loadBlob(d.crop_path);
        if (b) {
          const arc = `${folder}/crops/${prefix}_crop.jpg`;
          zip.file(arc, b);
          filesObj.crop = arc;
        }
      }
      if (payload.include_originals && d.frame_path) {
        const b = await loadBlob(d.frame_path);
        if (b) {
          const arc = `${folder}/originals/${prefix}_frame.jpg`;
          zip.file(arc, b);
          filesObj.original_frame = arc;
        }
      }

      capturesMeta.push({
        detection_id: d.id,
        source_file: d.media_filename,
        timestamp_sec: d.timestamp_sec,
        portrait_score: d.portrait_score,
        body_score: d.body_score,
        overall_score: d.overall_score,
        scene_summary: d.scene_summary,
        files: filesObj,
      });
    }

    zip.file(
      `${folder}/metadata.json`,
      JSON.stringify({ name: p.name, captures: capturesMeta }, null, 2)
    );
    manifest.push({ name: p.name, folder, captures_exported: capturesMeta.length });
  }

  zip.file('manifest.json', JSON.stringify({ persons: manifest }, null, 2));
  const outBlob = await zip.generateAsync({ type: 'blob' });
  const url = URL.createObjectURL(outBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'FriendFinder_GameAssets.zip';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function reassignBrowserDetection(
  detId: string,
  payload: { person_id?: string; new_person_name?: string }
): Promise<Detection> {
  const det = await getStoredDetection(detId);
  if (!det) throw new Error('Detección no encontrada');

  const oldPersonId = det.person_id;
  let targetPersonId = payload.person_id;

  if (!targetPersonId) {
    const allPersons = await getAllStoredPersons();
    const newName = (payload.new_person_name || '').trim() || `Persona ${allPersons.length + 1}`;
    const now = new Date().toISOString();
    const newPerson: StoredPerson = {
      id: uid(),
      name: newName,
      notes: '',
      centroid: det.embedding ? [...det.embedding] : null,
      cover_detection_id: det.id,
      best_portrait_id: det.id,
      best_body_id: det.has_cutout ? det.id : null,
      is_ignored: 0,
      created_at: now,
      updated_at: now,
    };
    await savePerson(newPerson);
    targetPersonId = newPerson.id;
  }

  const updated: StoredDetection = {
    ...det,
    person_id: targetPersonId,
    status: 'kept',
    discard_reason: null,
  };
  await saveDetection(updated);

  if (oldPersonId && oldPersonId !== targetPersonId) {
    await recomputeBrowserPersonStats(oldPersonId);
  }
  await recomputeBrowserPersonStats(targetPersonId);
  const { embedding: _e, ...clean } = updated;
  return clean;
}

/**
 * Global quality-first re-clustering across all stored detections with:
 * - Calibrated dlib 128-D distance threshold (0.54 at default) to unite the same person across angles/lighting.
 * - Per-frame Cannot-Link constraints (two faces in the SAME photo/frame can never merge into one person).
 * - Multi-anchor preservation of user-named friends & manual merges.
 * - Hierarchical cluster-to-cluster post-merge pass that reunites fragmented clusters via denoised centroids.
 */
export async function reclusterAllBrowserPersons(): Promise<{
  persons_before: number;
  persons_after: number;
  reassigned_count: number;
}> {
  const [settings, existingPersons, allDets] = await Promise.all([
    getBrowserSettings(),
    getAllStoredPersons(),
    getAllStoredDetections(),
  ]);
  const personsBefore = existingPersons.length;
  if (allDets.length === 0) {
    return { persons_before: personsBefore, persons_after: personsBefore, reassigned_count: 0 };
  }

  const threshold = Number(settings.similarity_threshold || '0.45');
  const maxDist = Math.max(0.42, Math.min(0.62, 0.54 - (threshold - 0.45) * 0.45));
  const detById = new Map(allDets.map((d) => [d.id, d]));

  const frameKeyOf = (d: StoredDetection): string | null =>
    d.face_bbox ? `${d.media_id}:${d.frame_idx}` : null;

  const computeMeanVec = (vecs: number[][]): number[] => {
    const dim = vecs[0].length;
    const mean = new Array<number>(dim).fill(0);
    for (const v of vecs) {
      for (let i = 0; i < dim; i++) mean[i] += v[i];
    }
    return mean.map((val) => val / vecs.length);
  };

  interface Cluster {
    id: string;
    name: string;
    notes: string;
    isNamed: boolean;
    centroid: number[];
    anchors: number[][];
    detIds: string[];
    frameKeys: Set<string>;
  }

  const clusters: Cluster[] = [];
  const assignedDetIds = new Set<string>();

  // 1. Seed user-named persons (and saved photo-less identities) with up to 8 compatible detections
  //    so user manual merges & multi-angle named profiles stay intact!
  const isCustomName = (n: string) => !/^Persona \d+$/i.test(n.trim());
  for (const p of existingPersons) {
    const mine = allDets
      .filter((d) => d.person_id === p.id && Array.isArray(d.embedding) && d.embedding.length > 0)
      .sort(
        (a, b) =>
          b.portrait_score + 0.35 * b.frontality_score - (a.portrait_score + 0.35 * a.frontality_score)
      );
    const isPhotoLessSaved = mine.length === 0 && Array.isArray(p.centroid) && p.centroid.length > 0;
    if (!isCustomName(p.name) && !isPhotoLessSaved) continue;

    const coverDet = p.cover_detection_id ? detById.get(p.cover_detection_id) : undefined;
    const bestDet =
      coverDet && Array.isArray(coverDet.embedding) && coverDet.embedding.length > 0
        ? coverDet
        : mine[0];

    if (bestDet && bestDet.embedding) {
      const fKeys = new Set<string>();
      const bestFk = frameKeyOf(bestDet);
      if (bestFk) fKeys.add(bestFk);

      const anchors: number[][] = [[...bestDet.embedding]];
      const detIds: string[] = [bestDet.id];
      assignedDetIds.add(bestDet.id);

      // Keep other compatible detections already belonging to this named person (preserves manual merges
      // while still allowing obvious unrelated false-merges > maxDist + 0.08 to split cleanly)
      for (const extra of mine) {
        if (extra.id === bestDet.id || !extra.embedding) continue;
        const fk = frameKeyOf(extra);
        if (fk && fKeys.has(fk)) continue;
        const distToCover = euclideanDistance(extra.embedding, bestDet.embedding);
        if (distToCover <= maxDist + 0.08) {
          detIds.push(extra.id);
          assignedDetIds.add(extra.id);
          if (fk) fKeys.add(fk);
          if (anchors.length < 8) {
            anchors.push([...extra.embedding]);
          }
        }
      }

      clusters.push({
        id: p.id,
        name: p.name,
        notes: p.notes || '',
        isNamed: true,
        centroid: computeMeanVec(anchors),
        anchors,
        detIds,
        frameKeys: fKeys,
      });
    } else if (Array.isArray(p.centroid) && p.centroid.length > 0) {
      const savedRefs =
        Array.isArray(p.reference_embeddings) && p.reference_embeddings.length > 0
          ? p.reference_embeddings.map((v) => [...v])
          : [[...p.centroid]];
      clusters.push({
        id: p.id,
        name: p.name,
        notes: p.notes || '',
        isNamed: true,
        centroid: [...p.centroid],
        anchors: savedRefs,
        detIds: [],
        frameKeys: new Set<string>(),
      });
    }
  }

  // 2. Sort remaining detections by quality + frontality DESC so crisp shots build pure clusters first
  const remaining = allDets
    .filter((d) => !assignedDetIds.has(d.id) && Array.isArray(d.embedding) && d.embedding.length > 0)
    .sort(
      (a, b) =>
        b.portrait_score + 0.35 * b.frontality_score - (a.portrait_score + 0.35 * a.frontality_score)
    );

  let autoCounter = 1;
  const nextAutoName = () => {
    const used = new Set(clusters.map((c) => c.name));
    while (used.has(`Persona ${autoCounter}`)) autoCounter++;
    return `Persona ${autoCounter++}`;
  };

  for (const det of remaining) {
    const emb = det.embedding!;
    const fk = frameKeyOf(det);
    const isVeryLowQuality =
      det.det_score < 0.45 || det.portrait_score < 32 || det.frontality_score < 30;
    const isCleanAnchor =
      det.det_score >= 0.52 && det.portrait_score >= 42 && det.frontality_score >= 38;
    const localMaxDist = isVeryLowQuality ? Math.max(0.40, maxDist - 0.025) : maxDist;

    let bestCluster: Cluster | null = null;
    let bestCombined = Infinity;
    let bestExMin = Infinity;
    let secondBestCombined = Infinity;

    for (const c of clusters) {
      // Cannot-Link: never place two detections from the same photo/video frame into the same person
      if (fk && c.frameKeys.has(fk)) continue;

      const cDist = euclideanDistance(emb, c.centroid);
      const anchorDists = c.anchors.map((a) => euclideanDistance(emb, a)).sort((a, b) => a - b);
      const bestEx = anchorDists[0];
      const topK = anchorDists.slice(0, Math.min(3, anchorDists.length));
      const avgTopDist = topK.reduce((s, v) => s + v, 0) / topK.length;
      const combined = 0.45 * cDist + 0.35 * avgTopDist + 0.20 * bestEx;

      const directMatch =
        cDist <= localMaxDist + 0.03 && avgTopDist <= localMaxDist + 0.02;
      const strongExemplarMatch =
        bestEx <= localMaxDist - 0.05 && cDist <= localMaxDist + 0.06;

      if (directMatch || strongExemplarMatch) {
        if (combined < bestCombined) {
          secondBestCombined = bestCombined;
          bestCombined = combined;
          bestExMin = bestEx;
          bestCluster = c;
        } else if (combined < secondBestCombined) {
          secondBestCombined = combined;
        }
      }
    }

    const isBorderline = bestCombined > localMaxDist - 0.03;
    const isAmbiguousTie =
      isBorderline &&
      secondBestCombined < Infinity &&
      secondBestCombined - bestCombined < 0.03 &&
      bestExMin > localMaxDist - 0.04;

    if (bestCluster && bestCombined <= localMaxDist + 0.015 && !isAmbiguousTie) {
      bestCluster.detIds.push(det.id);
      if (fk) bestCluster.frameKeys.add(fk);
      if (isCleanAnchor && bestCluster.anchors.length < 10) {
        bestCluster.anchors.push([...emb]);
        bestCluster.centroid = computeMeanVec(bestCluster.anchors);
      }
    } else {
      const fKeys = new Set<string>();
      if (fk) fKeys.add(fk);
      clusters.push({
        id: uid(),
        name: nextAutoName(),
        notes: '',
        isNamed: false,
        centroid: [...emb],
        anchors: [[...emb]],
        detIds: [det.id],
        frameKeys: fKeys,
      });
    }
  }

  // 3. Post-clustering hierarchical cluster-to-cluster merge pass:
  //    Denoised cluster centroids often reveal that two fragmented clusters are the exact same person!
  const setsOverlap = (a: Set<string>, b: Set<string>): boolean => {
    const [small, big] = a.size <= b.size ? [a, b] : [b, a];
    for (const item of small) {
      if (big.has(item)) return true;
    }
    return false;
  };

  let mergedAny = true;
  while (mergedAny) {
    mergedAny = false;
    let bestPair: [number, number] | null = null;
    let bestPairDist = Infinity;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const a = clusters[i];
        const b = clusters[j];
        // Never auto-merge two separately user-named friends
        if (a.isNamed && b.isNamed) continue;
        // Cannot-Link: never merge two clusters that appeared together in the same photo/frame
        if (setsOverlap(a.frameKeys, b.frameKeys)) continue;

        const cDist = euclideanDistance(a.centroid, b.centroid);
        if (cDist > maxDist + 0.01) continue;

        let minLink = Infinity;
        let sumLink = 0;
        let pairCount = 0;
        for (const ea of a.anchors) {
          for (const eb of b.anchors) {
            const d = euclideanDistance(ea, eb);
            if (d < minLink) minLink = d;
            sumLink += d;
            pairCount++;
          }
        }
        const avgLink = pairCount > 0 ? sumLink / pairCount : cDist;

        if (
          cDist <= maxDist &&
          (avgLink <= maxDist + 0.04 || minLink <= maxDist - 0.05) &&
          cDist < bestPairDist
        ) {
          bestPairDist = cDist;
          bestPair = [i, j];
        }
      }
    }

    if (bestPair) {
      const [i, j] = bestPair;
      // Keep named cluster (or larger cluster) as the survivor
      const [keepIdx, dropIdx] =
        clusters[j].isNamed || (!clusters[i].isNamed && clusters[j].detIds.length > clusters[i].detIds.length)
          ? [j, i]
          : [i, j];
      const keep = clusters[keepIdx];
      const drop = clusters[dropIdx];

      keep.detIds.push(...drop.detIds);
      for (const fk of drop.frameKeys) keep.frameKeys.add(fk);
      for (const anc of drop.anchors) {
        if (keep.anchors.length < 10) keep.anchors.push(anc);
      }
      keep.centroid = computeMeanVec(keep.anchors);
      clusters.splice(dropIdx, 1);
      mergedAny = true;
    }
  }

  // 4. Persist new clusters cleanly & remove obsolete person rows
  const keptPersonIds = new Set(clusters.map((c) => c.id));
  for (const oldP of existingPersons) {
    if (!keptPersonIds.has(oldP.id)) {
      await savePerson({ ...oldP, is_ignored: 1 });
    }
  }

  let reassignedCount = 0;
  const now = new Date().toISOString();

  for (const c of clusters) {
    const existing = existingPersons.find((p) => p.id === c.id);
    await savePerson({
      id: c.id,
      name: c.name,
      notes: c.notes,
      centroid: c.centroid,
      reference_embeddings: c.anchors.slice(0, 8),
      cover_detection_id: existing?.cover_detection_id || c.detIds[0] || null,
      best_portrait_id: c.detIds[0] || null,
      best_body_id: null,
      is_ignored: 0,
      created_at: existing?.created_at || now,
      updated_at: now,
    });

    // Guarantee at least the best detection in this cluster is marked 'kept'
    const clusterDets = c.detIds
      .map((id) => detById.get(id))
      .filter((d): d is StoredDetection => Boolean(d))
      .sort((a, b) => b.overall_score - a.overall_score);

    const hasKept = clusterDets.some((d) => d.status === 'kept');
    for (let i = 0; i < clusterDets.length; i++) {
      const d = clusterDets[i];
      const changed = d.person_id !== c.id;
      if (changed) reassignedCount++;
      const promoteBest = !hasKept && i === 0;
      await saveDetection({
        ...d,
        person_id: c.id,
        status: promoteBest ? 'kept' : d.status,
        discard_reason: promoteBest ? null : d.discard_reason,
      });
    }
    await recomputeBrowserPersonStats(c.id);
  }

  // Clean out only truly obsolete unnamed person rows (NEVER delete preserved photo-less clusters)
  const refreshedDets = await getAllStoredDetections();
  const usedIds = new Set<string>(refreshedDets.map((d) => d.person_id).filter(Boolean) as string[]);
  for (const c of clusters) usedIds.add(c.id);
  for (const oldP of existingPersons) {
    if (!usedIds.has(oldP.id)) {
      await deleteBrowserPerson(oldP.id);
    }
  }

  return {
    persons_before: personsBefore,
    persons_after: clusters.length,
    reassigned_count: reassignedCount,
  };
}
