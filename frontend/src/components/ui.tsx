import { AlertTriangle, CircleDashed, Info, XCircle } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'
import type { ModelState } from '../types/api'

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ')
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-3xl">
      {eyebrow && <p className="mb-2 font-mono text-xs uppercase tracking-[0.14em] text-fg-subtle">{eyebrow}</p>}
      <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-[28px]">{title}</h1>
      {children && <div className="mt-3 text-[15px] leading-relaxed text-fg-muted">{children}</div>}
    </header>
  )
}

export function Panel({ title, actions, children, className }: {
  title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={cx('rounded-lg border border-ink-700 bg-ink-900', className)}>
      {(title || actions) && (
        <div className="flex items-center justify-between gap-3 border-b border-ink-700 px-4 py-3">
          {title && <h2 className="text-sm font-medium text-fg">{title}</h2>}
          {actions}
        </div>
      )}
      <div className="p-4">{children}</div>
    </section>
  )
}

export function Stat({ label, value, hint, pending }: { label: string; value: ReactNode; hint?: ReactNode; pending?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-fg-subtle">{label}</dt>
      <dd className={cx('mt-1 font-mono text-lg tabular', pending ? 'text-pending' : 'text-fg')}>{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-fg-subtle">{hint}</p>}
    </div>
  )
}

const tone = {
  info: { icon: Info, cls: 'border-ink-600 bg-ink-850 text-fg-muted', ic: 'text-fg-muted' },
  pending: { icon: CircleDashed, cls: 'border-pending/30 bg-pending/5 text-fg-muted', ic: 'text-pending' },
  warn: { icon: AlertTriangle, cls: 'border-pending/40 bg-pending/10 text-fg', ic: 'text-pending' },
  error: { icon: XCircle, cls: 'border-danger/40 bg-danger/10 text-fg', ic: 'text-danger' },
}

export function Callout({ kind = 'info', title, children, action }: {
  kind?: keyof typeof tone; title?: string; children?: ReactNode; action?: ReactNode
}) {
  const t = tone[kind]
  const Icon = t.icon
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className={cx('flex gap-3 rounded-md border px-3.5 py-3 text-sm', t.cls)}>
      <Icon aria-hidden className={cx('mt-0.5 size-4 shrink-0', t.ic)} />
      <div className="min-w-0 flex-1">
        {title && <p className="font-medium text-fg">{title}</p>}
        {children && <div className={cx('leading-relaxed', title && 'mt-1')}>{children}</div>}
        {action && <div className="mt-2.5">{action}</div>}
      </div>
    </div>
  )
}

export function Button({ variant = 'primary', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost'
}) {
  const styles = {
    primary: 'bg-accent text-ink-950 hover:bg-accent/90 disabled:bg-ink-700 disabled:text-fg-subtle',
    secondary: 'border border-ink-600 bg-ink-850 text-fg hover:border-ink-500 hover:bg-ink-800 disabled:text-fg-subtle',
    ghost: 'text-fg-muted hover:bg-ink-800 hover:text-fg disabled:text-fg-subtle',
  }[variant]
  return (
    <button
      type="button"
      {...props}
      className={cx('inline-flex h-9 items-center justify-center gap-2 rounded-md px-3.5 text-sm font-medium transition-colors disabled:cursor-not-allowed', styles, className)}
    />
  )
}

const stateLabel: Record<ModelState, { label: string; cls: string }> = {
  ready: { label: 'Model ready', cls: 'bg-ok' },
  loading: { label: 'Loading model', cls: 'bg-pending animate-pulse' },
  not_configured: { label: 'No model configured', cls: 'bg-pending' },
  error: { label: 'Model error', cls: 'bg-danger' },
}

export function ModelStatePill({ state, compact }: { state: ModelState | 'offline'; compact?: boolean }) {
  const s = state === 'offline' ? { label: 'API offline', cls: 'bg-danger' } : stateLabel[state]
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-ink-700 bg-ink-900 px-2.5 py-1 text-xs text-fg-muted" title={s.label}>
      <span aria-hidden className={cx('size-1.5 rounded-full', s.cls)} />
      <span className={cx(compact && 'sr-only sm:not-sr-only')}>{s.label}</span>
    </span>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx('animate-pulse rounded bg-ink-800', className)} />
}

export function Pending({ children = 'Pending' }: { children?: ReactNode }) {
  return <span className="font-mono text-pending">{children}</span>
}

export const fmt = {
  pct: (v: number | null | undefined, digits = 1) => (v == null ? null : `${(v * 100).toFixed(digits)}%`),
  int: (v: number | null | undefined) => (v == null ? null : v.toLocaleString('en-US')),
  millions: (v: number | null | undefined) => (v == null ? null : `${(v / 1e6).toFixed(1)}M`),
  sec: (v: number | null | undefined) => (v == null ? null : v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(2)} s`),
}
