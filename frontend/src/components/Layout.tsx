import { NavLink, Outlet } from 'react-router-dom'
import { useApi } from '../hooks/useApi'
import { api, STATIC_DEMO } from '../services/api'
import { cx, ModelStatePill } from './ui'

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/recognize', label: 'Recognize' },
  { to: '/research', label: 'Research' },
  { to: '/model', label: 'Model' },
  { to: '/methodology', label: 'Methodology' },
]

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <svg aria-hidden viewBox="0 0 32 32" className="size-6">
        <rect width="32" height="32" rx="7" className="fill-ink-800" />
        <path d="M7 17c3 4 15 4 18 0" fill="none" className="stroke-accent" strokeWidth="2.4" strokeLinecap="round" />
        <path d="M7 15c3-3 15-3 18 0" fill="none" className="stroke-accent" strokeOpacity=".45" strokeWidth="2.4" strokeLinecap="round" />
      </svg>
      <span className="text-[15px] font-semibold tracking-tight text-fg">SilentVoice</span>
    </span>
  )
}

export default function Layout() {
  const health = useApi(api.health, 5000)
  const state = health.error ? 'offline' : health.data?.model_state
  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-ink-800 focus:px-3 focus:py-2">
        Skip to content
      </a>
      <header className="sticky top-0 z-20 border-b border-ink-700 bg-ink-950/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4 sm:gap-6 sm:px-6">
          <NavLink to="/" aria-label="SilentVoice home" className="shrink-0"><Wordmark /></NavLink>
          <nav aria-label="Main" className="-mx-1 flex min-w-0 flex-1 overflow-x-auto">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) => cx(
                  'shrink-0 rounded-md px-2.5 py-1.5 text-sm transition-colors',
                  isActive ? 'bg-ink-800 text-fg' : 'text-fg-muted hover:text-fg',
                )}
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          {state && <ModelStatePill state={state} compact />}
        </div>
      </header>
      {STATIC_DEMO && (
        <div role="note" className="border-b border-ink-700 bg-ink-900">
          <p className="mx-auto max-w-6xl px-4 py-2 text-xs text-fg-muted sm:px-6">
            Static demo: research numbers are the recorded pilot run (GRID, 500 unseen-speaker test clips).
            Live recognition needs the GPU backend:{' '}
            <a className="underline hover:text-fg" href="https://github.com/ame-ml-ai/silentvoice">run it locally</a>.
          </p>
        </div>
      )}
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
        <Outlet />
      </main>
      <footer className="border-t border-ink-700">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2 px-4 py-4 text-xs text-fg-subtle sm:px-6">
          <span>SilentVoice · university ML research prototype</span>
          <span>
            Model code:{' '}
            <a className="underline decoration-ink-500 underline-offset-2 hover:text-fg" href="https://github.com/mpc001/auto_avsr" target="_blank" rel="noreferrer">
              Auto-AVSR
            </a>{' '}
            (Ma et al., 2023), Apache-2.0
          </span>
        </div>
      </footer>
    </div>
  )
}
