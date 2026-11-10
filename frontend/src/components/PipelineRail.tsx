import { Check, Circle, Loader2, Minus, X } from 'lucide-react'
import type { Job, JobKind } from '../types/api'
import { cx } from './ui'

type StepState = 'todo' | 'active' | 'done' | 'failed' | 'skipped'

interface Step { label: string; detail: string; state: StepState; skipReason?: string }

export function pipelineSteps(kind: JobKind | null, job: Job | null, submitting: boolean, modelReady: boolean): Step[] {
  const s: Step[] = [
    { label: 'Validate', detail: 'decode video stream, resample to 25 fps', state: 'todo' },
    { label: 'Mouth region', detail: 'MediaPipe face keypoints, alignment, 96×96 crop', state: 'todo' },
    { label: 'Model', detail: '3D-CNN + ResNet-18 → Conformer → CTC/attention beam search', state: 'todo' },
    { label: 'Text', detail: 'SentencePiece tokens → transcript', state: 'todo' },
  ]
  if (submitting) s[0].state = 'active'
  if (!job) return s
  s[0].state = 'done'
  const recognizing = (job.stage ?? '').startsWith('Running visual speech')
  const skipReason = modelReady ? 'not run: preprocessing only' : 'not run: no model loaded'
  s[2].skipReason = s[3].skipReason = skipReason
  if (job.status === 'queued' || job.status === 'running') {
    s[1].state = recognizing ? 'done' : 'active'
    if (recognizing) s[2].state = 'active'
  } else if (job.status === 'completed') {
    s[1].state = 'done'
    s[2].state = kind === 'recognize' ? 'done' : 'skipped'
    s[3].state = kind === 'recognize' ? 'done' : 'skipped'
  } else {
    const failedAt = job.mouth_preview_png ? 2 : 1
    s[failedAt].state = 'failed'
    if (failedAt === 2) s[1].state = 'done'
  }
  return s
}

const icon: Record<StepState, typeof Check> = { todo: Circle, active: Loader2, done: Check, failed: X, skipped: Minus }

export default function PipelineRail({ steps }: { steps: Step[] }) {
  return (
    <ol className="grid gap-px overflow-hidden rounded-md border border-ink-700 bg-ink-700 sm:grid-cols-2 xl:grid-cols-4" aria-label="Processing pipeline">
      {steps.map((st, i) => {
        const Icon = icon[st.state]
        return (
          <li key={st.label} className="bg-ink-900 px-3 py-2.5" aria-current={st.state === 'active' ? 'step' : undefined}>
            <div className="flex items-center gap-2">
              <span className={cx(
                'flex size-5 items-center justify-center rounded-full border text-[10px]',
                st.state === 'done' && 'border-accent/50 bg-accent/10 text-accent',
                st.state === 'active' && 'border-accent text-accent',
                st.state === 'failed' && 'border-danger/60 bg-danger/10 text-danger',
                (st.state === 'todo' || st.state === 'skipped') && 'border-ink-600 text-fg-subtle',
              )}>
                <Icon aria-hidden className={cx('size-3', st.state === 'active' && 'animate-spin')} />
              </span>
              <span className="font-mono text-[11px] text-fg-subtle">{String(i + 1).padStart(2, '0')}</span>
              <span className={cx('text-sm', st.state === 'todo' || st.state === 'skipped' ? 'text-fg-subtle' : 'text-fg')}>
                {st.label}
              </span>
              <span className="sr-only">: {st.state}</span>
            </div>
            <p className="mt-1 text-[11px] leading-snug text-fg-subtle">{st.state === 'skipped' ? st.skipReason : st.detail}</p>
          </li>
        )
      })}
    </ol>
  )
}
