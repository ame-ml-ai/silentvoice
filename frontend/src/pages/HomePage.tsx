import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Callout, fmt, ModelStatePill, Panel, Pending } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

export default function HomePage() {
  const health = useApi(api.health)
  const info = useApi(api.modelInfo)
  const cmp = useApi(api.compare)

  return (
    <>
      <section className="mb-10 grid gap-8 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:items-end">
        <div>
          <p className="mb-3 font-mono text-xs uppercase tracking-[0.14em] text-fg-subtle">Visual speech recognition</p>
          <h1 className="max-w-2xl text-3xl font-semibold tracking-tight text-fg sm:text-4xl">
            SilentVoice turns lip movements into text, without listening.
          </h1>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-fg-muted">
            A research prototype that asks whether fine-tuning a pretrained lip-reading model on a new domain improves recognition
            under the limits of a single 8 GB laptop GPU. The recognizer only ever sees a cropped video of the mouth.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link to="/recognize" className="inline-flex h-10 items-center gap-2 rounded-md bg-accent px-4 text-sm font-medium text-ink-950 hover:bg-accent/90">
              Try recognition <ArrowRight aria-hidden className="size-4" />
            </Link>
            <Link to="/research" className="inline-flex h-10 items-center rounded-md border border-ink-600 px-4 text-sm text-fg hover:bg-ink-800">
              View research results
            </Link>
          </div>
        </div>
        <Panel title="System status" actions={health.data && <ModelStatePill state={health.data.model_state} />}>
          {health.error ? (
            <Callout kind="error" title="API unreachable">{health.error.message}</Callout>
          ) : (
            <dl className="space-y-2.5 text-sm">
              <div className="flex justify-between gap-4"><dt className="text-fg-subtle">Inference device</dt><dd className="font-mono text-fg-muted">{health.data?.device ?? '…'}</dd></div>
              <div className="flex justify-between gap-4"><dt className="text-fg-subtle">CUDA available</dt><dd className="font-mono text-fg-muted">{health.data ? String(health.data.cuda_available) : '…'}</dd></div>
              <div className="flex justify-between gap-4"><dt className="text-fg-subtle">Checkpoint</dt><dd className="truncate font-mono text-fg-muted">{info.data?.checkpoint ?? <Pending>none loaded</Pending>}</dd></div>
              <div className="flex justify-between gap-4"><dt className="text-fg-subtle">Experiments</dt><dd className="font-mono">{cmp.data?.status === 'complete' ? <span className="text-ok">complete</span> : <Pending>pending GPU runs</Pending>}</dd></div>
            </dl>
          )}
        </Panel>
      </section>

      <div className="grid gap-px overflow-hidden rounded-lg border border-ink-700 bg-ink-700 md:grid-cols-3">
        {[
          {
            k: 'Input', t: 'Visual only',
            d: 'Frames are decoded from the video stream alone, the face is aligned and a 96×96 mouth crop is extracted. Audio in the file is never read; tests verify that changing it changes nothing.',
          },
          {
            k: 'Model', t: 'Auto-AVSR (Ma et al., 2023)',
            d: `3D-CNN + ResNet-18 frontend, 12-block Conformer encoder, Transformer decoder, joint CTC/attention decoding. ${fmt.int(info.data?.parameters?.total) ?? '250M'} parameters, pretrained on LRS3 + VoxCeleb2.`,
          },
          {
            k: 'Data', t: 'GRID corpus, unseen speakers',
            d: 'Fine-tuning on 27 GRID talkers, model selection on 2 more, testing on 4 never-seen talkers (s1, s2, s20, s22). LRS3 is supported when access is granted.',
          },
        ].map((c) => (
          <div key={c.k} className="bg-ink-900 p-5">
            <p className="font-mono text-[11px] uppercase tracking-wider text-fg-subtle">{c.k}</p>
            <h2 className="mt-2 text-[15px] font-medium text-fg">{c.t}</h2>
            <p className="mt-2 text-sm leading-relaxed text-fg-muted">{c.d}</p>
          </div>
        ))}
      </div>
    </>
  )
}
