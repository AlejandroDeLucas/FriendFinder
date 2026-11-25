export interface Detection {
  id: string;
  media_id: string;
  person_id: string | null;
  person_name?: string | null;
  media_filename?: string;
  media_type?: 'image' | 'video';
  source_deleted?: boolean;
  timestamp_sec: number;
  frame_idx: number;
  face_bbox: [number, number, number, number] | null;
  body_bbox: [number, number, number, number] | null;
  det_score: number;
  portrait_score: number;
  body_score: number;
  overall_score: number;
  sharpness_score: number;
  frontality_score: number;
  separation_score: number;
  has_cutout: boolean;
  is_full_body: boolean;
  frame_path: string | null;
  portrait_path: string | null;
  cutout_path: string | null;
  crop_path: string | null;
  thumb_path: string | null;
  status: 'kept' | 'discarded';
  keep_category: string | null;
  discard_reason: string | null;
  scene_summary: string | null;
  scene_tags: string[] | null;
  scene_backend: string | null;
  created_at: string;
}

export interface PersonSummary {
  id: string;
  name: string;
  notes: string;
  tags?: string[];
  cover_detection_id: string | null;
  best_portrait_id: string | null;
  best_body_id: string | null;
  is_ignored: number;
  created_at: string;
  updated_at: string;
  total_detections: number;
  kept_count: number;
  discarded_count: number;
  cutout_count: number;
  max_portrait_score: number;
  max_body_score: number;
  media_count: number;
  cover_detection: Detection | null;
  best_portrait: Detection | null;
  best_body: Detection | null;
  similarity?: number;
}

export interface MediaAppearanceTimestamp {
  detection_id: string;
  timestamp_sec: number;
  frame_idx: number;
  portrait_score: number;
  body_score?: number;
  similarity?: number;
  status: 'kept' | 'discarded';
  thumb_path: string | null;
  portrait_path?: string | null;
  person_id?: string | null;
  person_name?: string | null;
}

export interface MediaAppearance {
  media_id: string;
  filename: string;
  media_type: 'image' | 'video';
  source_deleted: boolean;
  duration_sec?: number | null;
  appearances_count: number;
  best_score?: number;
  best_similarity?: number;
  timestamps: MediaAppearanceTimestamp[];
}

export interface PersonDetail extends PersonSummary {
  kept_detections: Detection[];
  discarded_detections: Detection[];
  media_appearances: MediaAppearance[];
}

export interface MediaItem {
  id: string;
  original_filename: string;
  media_type: 'image' | 'video';
  file_path: string;
  thumb_path: string | null;
  file_size: number;
  width: number;
  height: number;
  duration_sec: number | null;
  fps: number | null;
  frames_analyzed: number;
  detections_count: number;
  kept_count: number;
  discarded_count: number;
  source_deleted: number;
  keep_forever: number;
  expires_at: string | null;
  status: string;
  verdict: 'useful' | 'discarded' | 'no_people' | 'pending';
  verdict_reason: string | null;
  created_at: string;
  persons?: Array<{ id: string; name: string }>;
  detections?: Detection[];
}

export interface JobSummaryMediaResult {
  media_id: string;
  filename: string;
  media_type: 'image' | 'video';
  verdict: string;
  verdict_reason: string;
  detections: number;
  kept: number;
  discarded: number;
}

export interface JobInfo {
  id: string;
  job_type: string;
  status: 'queued' | 'running' | 'done' | 'error';
  progress: number;
  total_items: number;
  processed_items: number;
  current_step: string;
  error: string | null;
  summary: {
    total_media: number;
    useful_media: number;
    discarded_media: number;
    no_people_media: number;
    total_frames_analyzed: number;
    total_detections: number;
    kept_detections: number;
    discarded_detections: number;
    new_persons: number;
    media_results: JobSummaryMediaResult[];
    detected_persons?: Array<{
      id: string;
      name: string;
      is_new: boolean;
      thumb_path: string | null;
      captures_in_batch: number;
    }>;
  } | null;
  created_at: string;
  updated_at: string;
}

export interface SearchResult {
  found_faces: number;
  selected_face_index: number;
  query_faces: Array<{
    index: number;
    bbox: [number, number, number, number];
    det_score: number;
  }>;
  matched_persons: PersonSummary[];
  matched_media: MediaAppearance[];
  message: string;
}

export interface AppSettings {
  video_sample_fps: string;
  similarity_threshold: string;
  min_quality_score: string;
  max_portraits_per_person: string;
  max_fullbody_per_person: string;
  min_video_gap_seconds: string;
  source_retention_minutes: string;
  auto_scene_analysis: string;
  scene_backend: 'heuristic' | 'blip' | 'ollama';
  ollama_url: string;
  ollama_model: string;
  yolo_model: string;
  insightface_pack: string;
}

export interface SystemStats {
  persons_count: number;
  photos_count: number;
  videos_count: number;
  kept_captures: number;
  cutouts_count: number;
  discarded_captures: number;
  active_jobs: JobInfo[];
  device: {
    torch_device: string;
    onnx_provider: string;
    onnx_providers: string;
  };
}
