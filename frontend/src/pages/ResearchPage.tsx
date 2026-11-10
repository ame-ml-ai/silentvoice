import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Callout, cx, fmt, PageHeader, Panel, Pending, Skeleton, Stat } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { api } from '../services/api'
import type { ExperimentRecord } from '../types/api'

const axis = { stroke: 'var(--color-ink-600)', tick: { fill: 'var(--color-fg-subtle)', fontSize: 11 } }
const tooltip = {
  contentStyle: { background: 'var(--color-ink-850)', border: '1px solid var(--color-ink-600)', borderRadius: 6, fontSize: 12 },
  labelStyle: { color: 'var(--color-fg)' },
}

function ChartPending({ children }: { children: string }) {
  return (
    <div className="flex h-56 items-center justify-center rounded-md border border-dashed border-ink-600 px-6 text-center text-sm text-fg-subtle">
      {children}
    </div>
  )
}

function StatusBadge({ status }: { status: ExperimentRecord['status'] }) {
  return (
    <span className={cx('rounded px-1.5 py-0.5 font-mono text-[11px]',
      status === 'complete' && 'bg-ok/10 text-ok',
      status === 'pending' && 'bg-pending/10 text-pending',
      status === 'smoke' && 'bg-ink-700 text-fg-muted')}>
      {status}
    </span>
  )
}

const dash = (v: string | null) => v ?? '—'

export default function ResearchPage() {
  const cmp = useApi(api.compare)
  const exps = useApi(api.experiments)
  const c = cmp.data
  const b = c?.baseline
  const f = c?.finetuned
  // a comparison is shown only when every number it needs actually exists
  const complete = c?.status === 'complete' && b?.status === 'complete' && f?.status === 'complete'
    && b.wer != null && b.cer != null && f.wer != null && f.cer != null
  const relChange = complete ? fmt.pct(c?.wer_relative_change) : null
  const absChange = complete && c?.wer_absolute_change != null ? `${(c.wer_absolute_change * 100).toFixed(1)} pts absolute` : undefined

  const barData = complete && b && f ? [
    { metric: 'WER', Baseline: (b.wer ?? 0) * 100, 'Fine-tuned': (f.wer ?? 0) * 100 },
    { metric: 'CER', Baseline: (b.cer ?? 0) * 100, 'Fine-tuned': (f.cer ?? 0) * 100 },
  ] : []

  return (
    <>
      <PageHeader eyebrow="Research dashboard" title="Baseline vs. fine-tuned">
        Pretrained Auto-AVSR (Experiment A) against the same checkpoint after partial fine-tuning on GRID training speakers
        (Experiment B), both scored on the identical unseen-speaker test split. Values appear only after the experiments have actually run.
      </PageHeader>

      {cmp.error && <Callout kind="error" title="Could not load results">{cmp.error.message}</Callout>}

      <Panel className="mb-6">
        {cmp.loading ? <Skeleton className="h-14" /> : (
          <dl className="grid grid-cols-2 gap-6 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Baseline WER" value={fmt.pct(b?.wer) ?? 'Pending'} pending={b?.status !== 'complete'} />
            <Stat label="Fine-tuned WER" value={fmt.pct(f?.wer) ?? 'Pending'} pending={f?.status !== 'complete'} />
            <Stat label="WER change (relative)" value={relChange ?? 'Pending'} pending={!relChange} hint={absChange} />
            <Stat label="Baseline CER" value={fmt.pct(b?.cer) ?? 'Pending'} pending={b?.status !== 'complete'} />
            <Stat label="Fine-tuned CER" value={fmt.pct(f?.cer) ?? 'Pending'} pending={f?.status !== 'complete'} />
            <Stat label="Test clips" value={fmt.int(b?.num_samples) ?? 'Pending'} pending={b?.status !== 'complete'} hint={b?.split ? `split: ${b.split}` : undefined} />
          </dl>
        )}
      </Panel>

      {c && !complete && (
        <div className="mb-6">
          <Callout kind="pending" title="Experiments not run yet">
            <ul className="list-disc space-y-0.5 pl-4">{c.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
            <p className="mt-2">Both runs are scheduled for the RTX 5070 machine. See <span className="font-mono text-fg">CONTINUE_ON_RTX5070.md</span> for the exact commands.</p>
          </Callout>
        </div>
      )}

      <div className="mb-6 grid gap-6 lg:grid-cols-2">
        <Panel title="Error rates on the test split (%)">
          {complete ? (
            <ResponsiveContainer width="100%" height={224}>
              <BarChart data={barData} barGap={6}>
                <CartesianGrid vertical={false} stroke="var(--color-ink-700)" />
                <XAxis dataKey="metric" {...axis} />
                <YAxis {...axis} unit="%" />
                <Tooltip {...tooltip} formatter={(v) => `${Number(v).toFixed(2)}%`} cursor={{ fill: 'var(--color-ink-800)' }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Baseline" fill="var(--color-chart-2)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="Fine-tuned" fill="var(--color-accent)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          ) : <ChartPending>Chart appears once both evaluations have written results.json.</ChartPending>}
        </Panel>
        <Panel title="Fine-tuning curves">
          {c && c.loss_curve.length > 0 ? (
            <ResponsiveContainer width="100%" height={224}>
              <LineChart data={c.loss_curve}>
                <CartesianGrid stroke="var(--color-ink-700)" />
                <XAxis dataKey="step" {...axis} />
                <YAxis {...axis} />
                <Tooltip {...tooltip} />
                <Line type="monotone" dataKey="loss" stroke="var(--color-accent)" dot={false} strokeWidth={1.5} name="train loss" isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : <ChartPending>Training loss and validation WER appear here from experiments/finetuning/train_log.jsonl.</ChartPending>}
          {c && c.val_curve.length > 0 && (
            <p className="mt-2 font-mono text-[11px] text-fg-subtle">
              val WER by epoch: {c.val_curve.map((v) => `${v.epoch}:${(v.val_wer * 100).toFixed(1)}%`).join('  ')}
            </p>
          )}
        </Panel>
      </div>

      <Panel title="All experiment records">
        {exps.loading ? <Skeleton className="h-24" /> : exps.data && (
          <>
            <div className="-mx-4 overflow-x-auto">
              <table className="w-full min-w-[760px] text-left text-sm">
                <thead className="text-[11px] uppercase tracking-wider text-fg-subtle">
                  <tr className="border-b border-ink-700">
                    {['Experiment', 'Status', 'Split', 'Clips', 'WER', 'CER', 'Latency', 'Peak GPU', 'Trainable', 'Device'].map((h) => (
                      <th key={h} className="px-4 py-2 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="font-mono text-xs text-fg-muted">
                  {exps.data.experiments.map((e) => (
                    <tr key={e.id} className="border-b border-ink-800 last:border-0">
                      <td className="px-4 py-2.5 text-fg">{e.id}</td>
                      <td className="px-4 py-2.5"><StatusBadge status={e.status} /></td>
                      <td className="px-4 py-2.5">{dash(e.split)}</td>
                      <td className="px-4 py-2.5">{dash(fmt.int(e.num_samples))}</td>
                      <td className="px-4 py-2.5">{e.status === 'pending' ? <Pending>—</Pending> : dash(fmt.pct(e.wer, 2))}</td>
                      <td className="px-4 py-2.5">{e.status === 'pending' ? <Pending>—</Pending> : dash(fmt.pct(e.cer, 2))}</td>
                      <td className="px-4 py-2.5">{dash(fmt.sec(e.latency_mean_s))}</td>
                      <td className="px-4 py-2.5">{e.peak_gpu_gb != null ? `${e.peak_gpu_gb} GB` : '—'}</td>
                      <td className="px-4 py-2.5">{dash(fmt.millions(e.parameters_trainable))}</td>
                      <td className="px-4 py-2.5">{dash(e.device)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs leading-relaxed text-fg-subtle">{exps.data.note}</p>
          </>
        )}
      </Panel>
    </>
  )
}
