// Mirrors backend/app/schemas/api.py

export type ModelState = 'not_configured' | 'loading' | 'ready' | 'error'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed'
export type JobKind = 'preprocess' | 'recognize'

export interface Health {
  status: 'ok'
  version: string
  model_state: ModelState
  device: string
  cuda_available: boolean
}

export interface ArchitectureStage {
  name: string
  detail: string
  parameters: number | null
}

export interface ModelInfo {
  state: ModelState
  message: string
  model_name: string
  checkpoint: string | null
  checkpoint_md5: string | null
  device: Record<string, unknown> | null
  decode: Record<string, number> | null
  parameters: Record<string, number> | null
  vocab_size: number | null
  architecture: ArchitectureStage[]
  upstream: { repository: string; commit: string; license: string }
  input_modality: 'video'
}

export interface VideoMeta {
  filename: string
  size_bytes: number
  width: number
  height: number
  source_fps: number
  duration_s: number
  output_frames: number
  resampled: boolean
}

export interface JobCreated {
  job_id: string
  kind: JobKind
  status: JobStatus
  video: VideoMeta
}

export interface Job extends JobCreated {
  stage: string
  created_at: string
  finished_at: string | null
  detection_rate: number | null
  mouth_preview_png: string | null
  preview_frames: number | null
  text: string | null
  timings: Record<string, number>
  model_checkpoint: string | null
  error: { code: string; message: string } | null
}

export interface ExperimentRecord {
  id: string
  kind: 'baseline' | 'finetuning' | 'ablation'
  status: 'complete' | 'smoke' | 'pending'
  path: string
  split: string | null
  num_samples: number | null
  wer: number | null
  cer: number | null
  latency_mean_s: number | null
  real_time_factor: number | null
  peak_gpu_gb: number | null
  parameters_total: number | null
  parameters_trainable: number | null
  checkpoint: string | null
  device: string | null
  training: Record<string, unknown> | null
}

export interface ExperimentsResponse {
  experiments: ExperimentRecord[]
  note: string
}

export interface CompareResponse {
  status: 'complete' | 'pending'
  reasons: string[]
  baseline: ExperimentRecord | null
  finetuned: ExperimentRecord | null
  wer_absolute_change: number | null
  wer_relative_change: number | null
  cer_absolute_change: number | null
  cer_relative_change: number | null
  loss_curve: { step: number; loss: number; lr: number }[]
  val_curve: { epoch: number; val_loss: number; val_wer: number }[]
}
