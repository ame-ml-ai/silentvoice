import { Camera, Circle, FileVideo, Square, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Button, Callout, cx } from './ui'

const ACCEPT = '.mp4,.mov,.webm,.avi,.mkv,.mpg,.m4v,video/*'
const MAX_RECORD_S = 10

/** Upload a file or record a short clip from the webcam (video track only, no microphone). */
export default function VideoInput({ onSelect, disabled }: { onSelect: (f: File) => void; disabled?: boolean }) {
  const [mode, setMode] = useState<'upload' | 'webcam'>('upload')
  return (
    <div>
      <div role="group" aria-label="Video source" className="mb-3 inline-flex rounded-md border border-ink-700 bg-ink-950 p-0.5">
        {(['upload', 'webcam'] as const).map((m) => (
          <button
            key={m}
            type="button"
            aria-pressed={mode === m}
            onClick={() => setMode(m)}
            className={cx('inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-sm',
              mode === m ? 'bg-ink-800 text-fg' : 'text-fg-muted hover:text-fg')}
          >
            {m === 'upload' ? <Upload aria-hidden className="size-3.5" /> : <Camera aria-hidden className="size-3.5" />}
            {m === 'upload' ? 'Upload' : 'Webcam'}
          </button>
        ))}
      </div>
      {mode === 'upload' ? <DropZone onSelect={onSelect} disabled={disabled} /> : <Recorder onSelect={onSelect} disabled={disabled} />}
    </div>
  )
}

function DropZone({ onSelect, disabled }: { onSelect: (f: File) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files[0]
        if (f && !disabled) onSelect(f)
      }}
      className={cx('flex flex-col items-center justify-center gap-3 rounded-md border border-dashed px-4 py-10 text-center transition-colors',
        over ? 'border-accent bg-accent/5' : 'border-ink-600 bg-ink-950')}
    >
      <FileVideo aria-hidden className="size-6 text-fg-subtle" />
      <div>
        <p className="text-sm text-fg">Drop a video of a speaking face, or choose a file</p>
        <p className="mt-1 text-xs text-fg-subtle">MP4, MOV, WebM, AVI, MKV or MPG · up to 50 MB and 30 s · face visible, frontal</p>
      </div>
      <Button variant="secondary" onClick={() => input.current?.click()} disabled={disabled}>Choose file</Button>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        tabIndex={-1}
        className="sr-only"
        aria-hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f && !disabled) onSelect(f)
          e.target.value = ''
        }}
      />
    </div>
  )
}

type RecStatus = 'idle' | 'starting' | 'live' | 'recording' | 'denied' | 'no-camera' | 'failed' | 'unsupported'

function Recorder({ onSelect, disabled }: { onSelect: (f: File) => void; disabled?: boolean }) {
  const video = useRef<HTMLVideoElement>(null)
  const stream = useRef<MediaStream | null>(null)
  const recorder = useRef<MediaRecorder | null>(null)
  const mounted = useRef(true)
  const onSelectRef = useRef(onSelect)
  const disabledRef = useRef(disabled)
  useEffect(() => {
    onSelectRef.current = onSelect
    disabledRef.current = disabled
  })
  const [status, setStatus] = useState<RecStatus>(
    typeof navigator.mediaDevices?.getUserMedia === 'function' && typeof MediaRecorder !== 'undefined' ? 'idle' : 'unsupported',
  )
  const [elapsed, setElapsed] = useState(0)

  const stopStream = () => {
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
  }

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      const rec = recorder.current
      if (rec) {
        rec.ondataavailable = null
        rec.onstop = null
        if (rec.state !== 'inactive') rec.stop()
      }
      stopStream()
    }
  }, [])

  useEffect(() => {
    if (status !== 'recording') return
    const t0 = Date.now()
    const t = window.setInterval(() => {
      const s = (Date.now() - t0) / 1000
      setElapsed(s)
      if (s >= MAX_RECORD_S && recorder.current?.state === 'recording') recorder.current.stop()
    }, 100)
    return () => window.clearInterval(t)
  }, [status])

  const start = async () => {
    if (status === 'starting') return
    stopStream()
    setStatus('starting')
    try {
      // audio: false - SilentVoice never captures the microphone
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false })
      if (!mounted.current) {
        s.getTracks().forEach((t) => t.stop())
        return
      }
      stream.current = s
      if (video.current) video.current.srcObject = s
      setStatus('live')
    } catch (e) {
      if (!mounted.current) return
      const name = e instanceof DOMException ? e.name : ''
      setStatus(name === 'NotAllowedError' || name === 'SecurityError' ? 'denied'
        : name === 'NotFoundError' || name === 'OverconstrainedError' ? 'no-camera' : 'failed')
    }
  }

  const record = () => {
    if (!stream.current) return
    let rec: MediaRecorder
    try {
      rec = new MediaRecorder(stream.current)
    } catch {
      setStatus('failed')
      return
    }
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data) }
    rec.onerror = () => { if (mounted.current) setStatus('failed') }
    rec.onstop = () => {
      if (!mounted.current) return
      setStatus('live')
      setElapsed(0)
      if (chunks.length === 0 || disabledRef.current) return
      const type = rec.mimeType || 'video/webm'
      const ext = type.includes('mp4') ? 'mp4' : 'webm'
      onSelectRef.current(new File(chunks, `webcam-${Date.now()}.${ext}`, { type }))
    }
    recorder.current = rec
    rec.start()
    setStatus('recording')
  }

  if (status === 'unsupported') {
    return <Callout kind="warn" title="Webcam recording is not supported in this browser">Upload a video file instead.</Callout>
  }
  if (status === 'denied' || status === 'no-camera' || status === 'failed') {
    const copy = {
      denied: ['Camera access was blocked', 'Allow camera access for this site in your browser settings, or upload a file instead. Only video is requested, never the microphone.'],
      'no-camera': ['No camera found', 'Connect a camera or upload a video file instead.'],
      failed: ['The camera could not be used', 'It may be in use by another application. Close it and try again, or upload a file.'],
    }[status]
    return (
      <Callout kind="warn" title={copy[0]} action={<Button variant="secondary" onClick={() => void start()}>Try again</Button>}>
        {copy[1]}
      </Callout>
    )
  }
  return (
    <div className="space-y-3">
      <div className="relative aspect-[4/3] overflow-hidden rounded-md border border-ink-700 bg-ink-950">
        <video ref={video} autoPlay muted playsInline className="size-full -scale-x-100 object-cover" aria-label="Webcam preview" />
        {(status === 'idle' || status === 'starting') && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center">
            <p className="max-w-xs text-sm text-fg-muted">Records video only. Face the camera in good light and speak a short sentence.</p>
            <Button variant="secondary" onClick={() => void start()} disabled={disabled || status === 'starting'}>
              <Camera aria-hidden className="size-4" />{status === 'starting' ? 'Starting…' : 'Enable camera'}
            </Button>
          </div>
        )}
        {status === 'recording' && (
          <span className="absolute left-2 top-2 inline-flex items-center gap-1.5 rounded bg-ink-950/80 px-2 py-1 font-mono text-xs text-danger">
            <Circle aria-hidden className="size-2.5 animate-pulse fill-danger" /> REC {elapsed.toFixed(1)} / {MAX_RECORD_S}s
          </span>
        )}
      </div>
      <p className="sr-only" role="status">{status === 'recording' ? 'Recording' : status === 'live' ? 'Camera on, not recording' : ''}</p>
      {(status === 'live' || status === 'recording') && (
        <div className="flex flex-wrap gap-2">
          {status === 'recording'
            ? <Button variant="secondary" onClick={() => recorder.current?.stop()}><Square aria-hidden className="size-3.5" />Stop</Button>
            : <Button variant="secondary" onClick={record} disabled={disabled}><Circle aria-hidden className="size-3.5 fill-danger text-danger" />Record clip</Button>}
          {status === 'live' && (
            <Button variant="ghost" onClick={() => { stopStream(); setStatus('idle') }}>Turn camera off</Button>
          )}
        </div>
      )}
    </div>
  )
}
