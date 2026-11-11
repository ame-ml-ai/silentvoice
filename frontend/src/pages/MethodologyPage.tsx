import type { ReactNode } from 'react'
import { PageHeader } from '../components/ui'

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20 border-t border-ink-700 py-7 first:border-0 first:pt-0">
      <h2 className="text-base font-semibold text-fg">{title}</h2>
      <div className="mt-3 space-y-3 text-[15px] leading-relaxed text-fg-muted">{children}</div>
    </section>
  )
}

const TOC = [
  ['problem', 'Problem'], ['challenge', 'Why it is hard'], ['question', 'Research question'], ['data', 'Dataset'],
  ['model', 'Model'], ['training', 'Training approach'], ['metrics', 'Metrics'], ['protocol', 'Evaluation protocol'],
  ['limits', 'Limitations'], ['team', 'Team & credits'],
]

export default function MethodologyPage() {
  return (
    <>
      <PageHeader eyebrow="About & methodology" title="How SilentVoice is built and evaluated" />
      <div className="grid gap-10 lg:grid-cols-[12rem_minmax(0,1fr)]">
        <nav aria-label="On this page" className="hidden lg:block">
          <ul className="sticky top-20 space-y-1.5 text-sm">
            {TOC.map(([id, label]) => (
              <li key={id}><a href={`#${id}`} className="text-fg-subtle hover:text-fg">{label}</a></li>
            ))}
          </ul>
        </nav>
        <article className="max-w-3xl">
          <Section id="problem" title="Problem">
            <p>Visual speech recognition (VSR, “lip reading”) predicts the words a person says from video of their mouth alone. It matters where
              audio is missing, noisy or private, and for people who cannot rely on sound.</p>
          </Section>
          <Section id="challenge" title="Why it is hard">
            <p>Many phonemes look identical on the lips (/p/, /b/ and /m/ form one <em>viseme</em>), so words such as “pat”, “bat” and “mat” are
              visually ambiguous. Tongue and voicing are mostly invisible, speakers differ in how much they articulate, and pose, lighting and
              resolution vary. State-of-the-art word error rates on LRS3 are still around 20%, far above audio recognition.</p>
          </Section>
          <Section id="question" title="Research question">
            <p>Can fine-tuning a pretrained visual speech recognition model improve recognition performance on a selected evaluation dataset under
              limited computational resources (one RTX 4060 Laptop GPU, 8 GB)?</p>
          </Section>
          <Section id="data" title="Dataset">
            <p><strong className="text-fg">GRID</strong> (Cooke et al., 2006): 33 talkers with usable video, 1,000 fixed-grammar sentences each
              (“bin blue at f two now”), 25 fps, freely available for research from Zenodo. None of the Auto-AVSR checkpoints were trained on it,
              so it provides a clean domain-shift target.</p>
            <p>Splits are <strong className="text-fg">speaker-disjoint</strong>: test s1, s2, s20, s22 (the LipNet unseen-speaker protocol),
              validation s3, s4, training the remaining 27 talkers. A test speaker never appears in training or model selection; this is
              checked automatically before training starts.</p>
            <p><strong className="text-fg">LRS3</strong> is supported by the preprocessing code but its distribution is restricted, and every
              available checkpoint was trained on its training portion. It is used only if authorised access is obtained.</p>
          </Section>
          <Section id="model" title="Model">
            <p>Auto-AVSR's visual-only model (Ma et al., ICASSP 2023): a 3D-convolution + ResNet-18 frontend, a 12-block Conformer encoder and a
              6-block Transformer decoder, trained with a hybrid 0.1·CTC + 0.9·attention loss and decoded with joint CTC/attention beam search.
              The pretrained checkpoint used for both experiments is <span className="font-mono text-fg">vsr_trlrs3vox2_base.pth</span>
              (LRS3 + VoxCeleb2, 1,759 h).</p>
          </Section>
          <Section id="training" title="Training approach">
            <p>Partial fine-tuning keeps the visual frontend and the lower 8 Conformer blocks frozen and trains the top 4 encoder blocks, the
              decoder and the CTC head (125.2M of 250.4M parameters). AdamW at 1e-5 with warmup and cosine decay, bf16 mixed precision,
              gradient accumulation and frame-budget batching keep memory inside 8 GB (measured peak: 2.96 GB). The checkpoint with the lowest validation WER is kept.</p>
          </Section>
          <Section id="metrics" title="Metrics">
            <p><strong className="text-fg">Word error rate</strong> = (S + D + I) / N, the substitutions, deletions and insertions needed to turn
              the hypothesis into the reference, divided by the number of reference words. <strong className="text-fg">Character error rate</strong>
              {' '}is the same at character level. Both are computed corpus-wide with <span className="font-mono text-fg">jiwer</span> after identical text normalisation.</p>
          </Section>
          <Section id="protocol" title="Evaluation protocol">
            <p>Baseline and fine-tuned models are evaluated by the same script, on exactly the same test clips, with the same decoding settings.
              The dashboard refuses to compare runs whose split or sample count differ. Every result file records the checkpoint hash, seed,
              software versions, hardware and command used.</p>
          </Section>
          <Section id="limits" title="Limitations">
            <ul className="list-disc space-y-1.5 pl-5">
              <li>GRID has a tiny, fixed vocabulary and grammar; gains there do not imply gains on open-vocabulary speech.</li>
              <li>Fine-tuning on GRID may reduce accuracy on general speech (catastrophic forgetting); this is not measured unless LRS3 is available.</li>
              <li>MediaPipe face detection replaces the RetinaFace detector used for the official LRS3 crops; small preprocessing differences remain.</li>
              <li>Live demos on new speakers, poses and lighting are out of distribution and will be less accurate than test-set numbers.</li>
              <li>Results come from a single seed on a single GPU; no confidence intervals unless additional runs are made.</li>
            </ul>
          </Section>
          <Section id="team" title="Team & credits">
            <p>Team details have not been supplied yet.</p>
            <p>Model and preprocessing code: Auto-AVSR by Pingchuan Ma et al. (Imperial College London), Apache-2.0. GRID corpus: Cooke, Barker,
              Cunningham and Shao (University of Sheffield). Face detection: Google MediaPipe.</p>
          </Section>
        </article>
      </div>
    </>
  )
}
