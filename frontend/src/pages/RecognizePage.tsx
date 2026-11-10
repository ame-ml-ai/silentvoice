import { Check, Copy, ScanFace, Sparkles, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import MouthStrip from '../components/MouthStrip'
import PipelineRail, { pipelineSteps } from '../components/PipelineRail'
import { Button, Callout, fmt, PageHeader, Panel } from '../components/ui'
import VideoInput from '../components/VideoInput'
import { useApi, useJob } from '../hooks/useApi'
import { api, ApiError } from '../services/api'
import type { JobKind } from '../types/api'

const HINTS: Record<string, string> = {
  no_face: 'Use a well-lit, roughly frontal clip where the whole face stays in frame.',
  invalid_video: 'Try re-exporting the clip as H.264 MP4.',
  video_too_long: 'Trim the clip to 30 seconds or less.',
  video_too_short: 'Record at least half a second of speech.',
  file_too_large: 'Compress the video or shorten it.',
  unsupported_format: 'Convert the file to MP4.',
  network_error: 'Start the API with: uvicorn backend.app.main:app --port 8000',
}

function ErrorCallout({ code, message }: { code: string; message: string }) {
  return (
    <Callout kind="error" title="Processing failed">
      {message}
      {HINTS[code] && <p className="mt-1 text-fg-muted">{HINTS[code]}</p>}
      <p className="mt-1 font-mono text-[11px] text-fg-subtle">code: {code}</p>
    </Callout>
  )
}

export default function RecognizePage() {
  const health = useApi(api.health, 4000)
  const modelState = health.data?.model_state
  const modelReady = modelState === 'ready'

  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => {
      URL.revokeObjectURL(url)
      setPreviewUrl(null)
    }
  }, [file])

  const [kind, setKind] = useState<JobKind | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<ApiError | null>(null)
  const { job, error: pollError } = useJob(jobId)
  const [copied, setCopied] = useState<'yes' | 'failed' | null>(null)
  const seq = useRef(0) // invalidates in-flight submissions when the input changes

  const busy = submitting || job?.status === 'queued' || job?.status === 'running'

  const choose = (f: File | null) => {
    if (busy) return
    seq.current += 1
    setFile(f)
    setJobId(null)
    setKind(null)
    setSubmitError(null)
  }

  const submit = async (k: JobKind) => {
    if (!file || busy) return
    const mine = ++seq.current
    setSubmitting(true)
    setSubmitError(null)
    setJobId(null)
    setKind(k)
    try {
      const created = await api.submit(k, file)
      if (mine === seq.current) setJobId(created.job_id)
    } catch (e) {
      if (mine !== seq.current) return
      setSubmitError(e instanceof ApiError ? e : new ApiError(0, 'unknown', String(e)))
      health.reload()
    } finally {
      if (mine === seq.current) setSubmitting(false)
    }
  }

  useEffect(() => {
    if (!copied) return
    const t = window.setTimeout(() => setCopied(null), 1500)
    return () => window.clearTimeout(t)
  }, [copied])

  const copy = async () => {
    if (!job?.text) return
    try {
      await navigator.clipboard.writeText(job.text)
      setCopied('yes')
    } catch {
      setCopied('failed')
    }
  }

  const err = submitError ?? pollError
  const steps = pipelineSteps(kind, job, submitting, modelReady)

  return (
    <>
      <PageHeader eyebrow="Live recognition" title="Read speech from lip movements">
        The model receives only the cropped mouth region. Any audio track in your file is never decoded.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="space-y-4">
          <Panel title="Input">
            <VideoInput onSelect={choose} disabled={busy} />
          </Panel>

          {file && previewUrl && (
            <Panel title="Preview" actions={
              <Button variant="ghost" className="h-7 px-2" onClick={() => choose(null)} aria-label="Remove video" disabled={busy}>
                <X aria-hidden className="size-4" />
              </Button>
            }>
              <video src={previewUrl} controls muted playsInline className="aspect-video w-full rounded bg-ink-950 object-contain" />
              <p className="mt-2 truncate font-mono text-[11px] text-fg-subtle" title={file.name}>
                {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB · preview muted
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Button onClick={() => void submit('recognize')} disabled={busy || !modelReady}>
                  <Sparkles aria-hidden className="size-4" />Recognize speech
                </Button>
                <Button variant="secondary" onClick={() => void submit('preprocess')} disabled={busy}>
                  <ScanFace aria-hidden className="size-4" />Extract mouth region only
                </Button>
              </div>
              {!modelReady && modelState && (
                <p className="mt-3 text-xs leading-relaxed text-fg-subtle">
                  {modelState === 'loading'
                    ? 'Recognition is disabled while the model loads.'
                    : 'Recognition is disabled: no Auto-AVSR checkpoint is loaded on this server. Mouth-region extraction still runs the real preprocessing pipeline.'}
                </p>
              )}
            </Panel>
          )}
        </div>

        <div className="space-y-4">
          <PipelineRail steps={steps} />

          <p className="sr-only" role="status" aria-live="polite">
            {submitting ? 'Uploading and validating video'
              : job?.status === 'completed' ? (job.kind === 'recognize' ? `Recognized: ${job.text || 'no text'}` : 'Mouth region extracted')
              : job?.status === 'failed' ? `Failed: ${job.error?.message ?? ''}`
              : job ? job.stage : ''}
          </p>
          <Panel title="Result">
            {!file && !job && !err && (
              <div className="py-10 text-center">
                <p className="text-sm text-fg-muted">Choose or record a video to begin.</p>
                <p className="mx-auto mt-1 max-w-sm text-xs text-fg-subtle">Short clips (2–10 s) of one frontal speaker work best. GRID-style sentences match the fine-tuning domain.</p>
              </div>
            )}
            {file && !job && !err && !submitting && (
              <p className="py-10 text-center text-sm text-fg-muted">Ready. Start recognition or extract the mouth region.</p>
            )}
            {submitting && <p className="py-10 text-center text-sm text-fg-muted">Uploading and validating…</p>}
            {err && <ErrorCallout code={err.code} message={err.message} />}

            {job && (
              <div className="space-y-5">
                {(job.status === 'queued' || job.status === 'running') && (
                  <p className="text-sm text-fg-muted">{job.stage}…</p>
                )}
                {job.status === 'failed' && job.error && <ErrorCallout code={job.error.code} message={job.error.message} />}

                {job.mouth_preview_png && (
                  <div>
                    <h3 className="mb-2 text-xs font-medium uppercase tracking-wider text-fg-subtle">What the model sees</h3>
                    <MouthStrip png={job.mouth_preview_png} detectionRate={job.detection_rate} frames={job.preview_frames ?? job.video.output_frames} />
                  </div>
                )}

                {job.status === 'completed' && job.kind === 'recognize' && (
                  <div>
                    <div className="mb-2 flex items-center justify-between">
                      <h3 className="text-xs font-medium uppercase tracking-wider text-fg-subtle">Recognized text</h3>
                      <Button variant="ghost" className="h-7 px-2 text-xs" onClick={() => void copy()} disabled={!job.text}>
                        {copied === 'yes' ? <Check aria-hidden className="size-3.5" /> : <Copy aria-hidden className="size-3.5" />}
                        {copied === 'yes' ? 'Copied' : copied === 'failed' ? 'Copy blocked' : 'Copy'}
                      </Button>
                    </div>
                    <p className="rounded-md border border-ink-700 bg-ink-950 px-4 py-4 font-mono text-lg leading-relaxed tracking-wide text-fg">
                      {job.text || <span className="text-fg-subtle">(empty hypothesis: the model emitted no tokens)</span>}
                    </p>
                    <p className="mt-2 font-mono text-[11px] text-fg-subtle">checkpoint: {job.model_checkpoint}</p>
                  </div>
                )}

                {job.status === 'completed' && job.kind === 'preprocess' && (
                  <Callout kind="pending" title="Transcript not produced">
                    {modelReady
                      ? 'You ran preprocessing only. Choose “Recognize speech” for a transcript.'
                      : 'Inference requires the configured Auto-AVSR model environment, which this server does not have yet. The crops above are real output of the preprocessing pipeline.'}
                  </Callout>
                )}

                {(job.status === 'completed' || job.status === 'failed') && (
                  <dl className="grid grid-cols-2 gap-x-4 gap-y-3 border-t border-ink-700 pt-4 sm:grid-cols-4">
                    <div><dt className="text-[11px] text-fg-subtle">Source</dt><dd className="font-mono text-xs text-fg-muted">{job.video.width}×{job.video.height} @ {job.video.source_fps} fps</dd></div>
                    <div><dt className="text-[11px] text-fg-subtle">Model input</dt><dd className="font-mono text-xs text-fg-muted">{job.video.output_frames} frames @ 25 fps{job.video.resampled ? ' (resampled)' : ''}</dd></div>
                    {Object.entries(job.timings).map(([k, v]) => (
                      <div key={k}><dt className="text-[11px] text-fg-subtle">{k.replace(/_s$/, '').replace('_', ' ')}</dt><dd className="font-mono text-xs text-fg-muted">{fmt.sec(v)}</dd></div>
                    ))}
                  </dl>
                )}
              </div>
            )}
          </Panel>
          {health.data && (
            <p className="text-xs text-fg-subtle">
              Server device: <span className="font-mono">{health.data.device}</span>
              {health.data.device === 'cpu' && ' · CPU timings are not representative of the RTX 5070.'}
            </p>
          )}
        </div>
      </div>
    </>
  )
}
