/** The real 96x96 mouth crops the backend produced: exactly what the model sees. */
export default function MouthStrip({ png, detectionRate, frames }: { png: string; detectionRate: number | null; frames: number }) {
  return (
    <figure>
      <div className="overflow-x-auto rounded-md border border-ink-700 bg-ink-950 p-1.5">
        <img
          src={`data:image/png;base64,${png}`}
          alt={`${Math.min(8, frames)} evenly spaced mouth-region crops extracted from the video`}
          className="pixelated block h-auto w-full min-w-[480px]"
        />
      </div>
      <figcaption className="mt-2 flex flex-wrap justify-between gap-2 font-mono text-[11px] text-fg-subtle">
        <span>{Math.min(8, frames)} of {frames} frames · 96×96 crop → 88×88 grayscale model input</span>
        {detectionRate != null && <span>face detected in {(detectionRate * 100).toFixed(0)}% of frames</span>}
      </figcaption>
    </figure>
  )
}
