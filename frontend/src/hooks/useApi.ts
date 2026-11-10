import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../services/api'
import type { Job } from '../types/api'

export interface AsyncState<T> {
  data: T | null
  error: ApiError | null
  loading: boolean
  reload: () => void
}

const asApiError = (e: unknown) => (e instanceof ApiError ? e : new ApiError(0, 'unknown', String(e)))

/** Fetch on mount (and on reload), optionally polling. Out-of-order and post-unmount responses are ignored. */
export function useApi<T>(fn: () => Promise<T>, pollMs?: number): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const fnRef = useRef(fn)
  const seq = useRef(0)
  const alive = useRef(true)
  useEffect(() => {
    fnRef.current = fn
  })

  const load = useCallback(async () => {
    const mine = ++seq.current
    try {
      const value = await fnRef.current()
      if (!alive.current || mine !== seq.current) return
      setData(value)
      setError(null)
    } catch (e) {
      if (!alive.current || mine !== seq.current) return
      setError(asApiError(e))
    } finally {
      if (alive.current && mine === seq.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    alive.current = true
    void load()
    const t = pollMs ? window.setInterval(() => void load(), pollMs) : undefined
    return () => {
      alive.current = false
      window.clearInterval(t)
    }
  }, [load, pollMs])

  return { data, error, loading, reload: () => void load() }
}

const MAX_POLL_FAILURES = 4

/** Poll a job until it completes or fails; transient network errors are retried with backoff. */
export function useJob(jobId: string | null) {
  const [state, setState] = useState<{ id: string | null; job: Job | null; error: ApiError | null }>({
    id: null, job: null, error: null,
  })

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined
    let failures = 0
    const tick = async () => {
      try {
        const j = await api.job(jobId)
        if (cancelled) return
        failures = 0
        setState({ id: jobId, job: j, error: null })
        if (j.status === 'queued' || j.status === 'running') timer = window.setTimeout(tick, 600)
      } catch (e) {
        if (cancelled) return
        const err = asApiError(e)
        failures += 1
        // a 404 is final (job evicted / server restarted); network blips are retried
        if (err.status === 404 || failures >= MAX_POLL_FAILURES) {
          setState((s) => ({ id: jobId, job: s.id === jobId ? s.job : null, error: err }))
        } else {
          timer = window.setTimeout(tick, 600 * 2 ** failures)
        }
      }
    }
    void tick()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [jobId])

  // ignore results that belong to a previous job id
  const current = state.id === jobId
  return { job: current ? state.job : null, error: current ? state.error : null }
}
