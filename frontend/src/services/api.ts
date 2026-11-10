import type { CompareResponse, ExperimentsResponse, Health, Job, JobCreated, JobKind, ModelInfo } from '../types/api'

export class ApiError extends Error {
  status: number
  code: string
  details: Record<string, unknown>

  constructor(status: number, code: string, message: string, details: Record<string, unknown> = {}) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

/** Built with VITE_STATIC_DEMO=1 (e.g. on Vercel): no backend, read-only endpoints are served
 *  from JSON exported by scripts/export_static_demo.py; recognition needs the local GPU API. */
export const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === '1'

const STATIC_FILES: Record<string, string> = {
  '/api/health': '/demo/health.json',
  '/api/model/info': '/demo/model-info.json',
  '/api/experiments': '/demo/experiments.json',
  '/api/experiments/compare': '/demo/compare.json',
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC_DEMO) {
    const file = STATIC_FILES[path]
    if (!file) {
      throw new ApiError(503, 'static_demo',
        'This hosted demo has no inference server: the 250M-parameter model needs a GPU backend. '
        + 'Clone the repository and run the API locally to recognise videos.')
    }
    path = file
  }
  let res: Response
  try {
    res = await fetch(path, init)
  } catch {
    throw new ApiError(0, 'network_error', 'Cannot reach the SilentVoice API. Start the backend on port 8000.')
  }
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const err = body?.error
    throw new ApiError(res.status, err?.code ?? 'http_error', err?.message ?? `Request failed (${res.status}).`, err?.details)
  }
  return body as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  modelInfo: () => request<ModelInfo>('/api/model/info'),
  experiments: () => request<ExperimentsResponse>('/api/experiments'),
  compare: () => request<CompareResponse>('/api/experiments/compare'),
  job: (id: string) => request<Job>(`/api/jobs/${encodeURIComponent(id)}`),
  submit: (kind: JobKind, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<JobCreated>(`/api/${kind}`, { method: 'POST', body: form })
  },
}
