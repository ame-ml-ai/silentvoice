import { ArrowDown } from 'lucide-react'
import { Callout, fmt, PageHeader, Panel, Skeleton, Stat } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'

const SHARE_COLORS: Record<string, string> = {
  frontend: 'var(--color-chart-2)',
  proj_encoder: 'var(--color-fg-subtle)',
  encoder: 'var(--color-accent)',
  decoder: 'var(--color-pending)',
  ctc: 'var(--color-ok)',
}
const SHARE_LABEL: Record<string, string> = {
  frontend: 'Visual frontend', proj_encoder: 'Projection', encoder: 'Conformer encoder', decoder: 'Transformer decoder', ctc: 'CTC head',
}

export default function ModelPage() {
  const { data, error, loading } = useApi(api.modelInfo)
  const p = data?.parameters
  const parts = p ? (['frontend', 'proj_encoder', 'encoder', 'decoder', 'ctc'] as const).map((k) => ({ k, n: p[k] })) : []

  return (
    <>
      <PageHeader eyebrow="Model explorer" title="Auto-AVSR visual speech recognizer">
        Every stage below is read from the model code pinned in <span className="font-mono text-fg">third_party/auto_avsr</span>; parameter
        counts are computed from that code, not quoted from a paper.
      </PageHeader>
      {error && <Callout kind="error" title="Could not load model information">{error.message}</Callout>}
      {loading && <Skeleton className="h-96" />}

      {data && (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          <Panel title="Data flow">
            <ol className="space-y-0">
              {data.architecture.map((s, i) => (
                <li key={s.name}>
                  <div className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-start gap-3 rounded-md border border-ink-700 bg-ink-850 px-3 py-3">
                    <span className="pt-0.5 font-mono text-[11px] text-fg-subtle">{String(i + 1).padStart(2, '0')}</span>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-fg">{s.name}</p>
                      <p className="mt-0.5 text-xs leading-relaxed text-fg-muted">{s.detail}</p>
                    </div>
                    <span className="pt-0.5 font-mono text-xs text-fg-muted">{s.parameters != null ? fmt.millions(s.parameters) : ''}</span>
                  </div>
                  {i < data.architecture.length - 1 && (
                    <div className="flex h-5 items-center pl-[1.15rem]"><ArrowDown aria-hidden className="size-3.5 text-ink-500" /></div>
                  )}
                </li>
              ))}
            </ol>
          </Panel>

          <div className="space-y-6">
            <Panel title="Parameters">
              {p && (
                <>
                  <p className="font-mono text-2xl text-fg tabular">{fmt.int(p.total)}</p>
                  <p className="mt-0.5 text-xs text-fg-subtle">total; matches the "250M" listed in the Auto-AVSR model zoo</p>
                  <div className="mt-4 flex h-2.5 overflow-hidden rounded-full bg-ink-800" role="img"
                    aria-label={parts.map((x) => `${SHARE_LABEL[x.k]} ${((x.n / p.total) * 100).toFixed(1)}%`).join(', ')}>
                    {parts.map((x) => <span key={x.k} style={{ width: `${(x.n / p.total) * 100}%`, background: SHARE_COLORS[x.k] }} />)}
                  </div>
                  <ul className="mt-3 space-y-1.5 text-xs">
                    {parts.map((x) => (
                      <li key={x.k} className="flex items-center justify-between gap-3">
                        <span className="flex items-center gap-2 text-fg-muted">
                          <span aria-hidden className="size-2 rounded-sm" style={{ background: SHARE_COLORS[x.k] }} />{SHARE_LABEL[x.k]}
                        </span>
                        <span className="font-mono text-fg-muted">{fmt.int(x.n)} · {((x.n / p.total) * 100).toFixed(1)}%</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </Panel>

            <Panel title="Loaded checkpoint">
              {data.state === 'ready' ? (
                <dl className="grid grid-cols-2 gap-4">
                  <Stat label="Checkpoint" value={<span className="text-sm">{data.checkpoint}</span>} />
                  <Stat label="MD5" value={<span className="text-sm">{data.checkpoint_md5?.slice(0, 10)}…</span>} />
                  <Stat label="Device" value={<span className="text-sm">{String(data.device?.name ?? data.device?.type)}</span>} />
                  <Stat label="Beam size" value={<span className="text-sm">{data.decode?.beam_size}</span>} />
                </dl>
              ) : (
                <Callout kind={data.state === 'error' ? 'error' : 'pending'} title={data.state === 'loading' ? 'Loading' : 'No checkpoint loaded'}>
                  {data.message}
                </Callout>
              )}
            </Panel>

            <Panel title="Provenance">
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between gap-3"><dt className="text-fg-subtle">Repository</dt>
                  <dd><a className="font-mono text-fg-muted underline decoration-ink-500 underline-offset-2 hover:text-fg" href={data.upstream.repository} target="_blank" rel="noreferrer">mpc001/auto_avsr</a></dd></div>
                <div className="flex justify-between gap-3"><dt className="text-fg-subtle">Commit</dt><dd className="font-mono text-fg-muted">{data.upstream.commit.slice(0, 12)}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-fg-subtle">License</dt><dd className="text-right text-fg-muted">{data.upstream.license}</dd></div>
                <div className="flex justify-between gap-3"><dt className="text-fg-subtle">Input modality</dt><dd className="font-mono text-fg-muted">{data.input_modality} only</dd></div>
              </dl>
            </Panel>
          </div>
        </div>
      )}
    </>
  )
}
