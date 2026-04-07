# FluidAudio What's New

## Upgrade: tag-20260403 → tag-20260406

### New Features

#### TTS: PocketTTS Session API (#471)
`PocketTtsSession` is a new actor for long-running streaming TTS. Previous behavior paid the full
voice prefill cost (~125 CoreML predictions) on every `synthesizeStreaming()` call, resetting Mimi
decoder state between utterances, causing latency and audio discontinuity.

The session performs voice prefill **once** at creation and accepts streamed text via `enqueue()`.
Each subsequent utterance only pays the text prefill cost. Mimi decoder state persists across
utterances for seamless audio continuity. Cancellation is awaitable: `await session.cancel()`
blocks until the Neural Engine is free, preventing inference loop stacking.

`AudioFrame` now includes `utteranceIndex` for text synchronization on the consumer side.

#### ASR: Japanese CTC Model (#478)
New `CtcJaManager` and `CtcJaModels` for Japanese speech recognition.
- **Model**: `FluidInference/parakeet-ctc-0.6b-ja-coreml`
- **Architecture**: 600M parameter CTC-only, 3,072 vocab SentencePiece tokens
- **Accuracy**: 6.5% CER on JSUT basic5000, 13.3% on Mozilla Common Voice 16.1
- **API**: Synchronous `transcribe(audio: [Float])` — model id `fa-parakeet-ctc-ja`

#### ASR: Mandarin Chinese CTC Model — Experimental (#476)
New `CtcZhCnManager` and `CtcZhCnModels` for Mandarin Chinese speech recognition.
- **Model**: `FluidInference/parakeet-ctc-0.6b-zh-cn-coreml` (int8: 571 MB, fp32: 1.1 GB)
- **Accuracy**: 8.23% mean CER on THCHS-30 (2,495 samples), 14.83x RTFx
- **API**: Synchronous `transcribe(audio: [Float])` — model id `fa-parakeet-ctc-zh-cn`
- **Note**: Experimental — API and accuracy may change in future releases

#### ASR: PunctuationCommitLayer (#466)
New `PunctuationCommitLayer` actor wrapping any streaming ASR engine for smart text segmentation.
Separates "committed" (finalized) text from "ghost" (speculative) text at sentence boundaries.
- Commits at `.`, `!`, `?`; configurable debounce timeout behavior
- Engine-agnostic: works with any `StreamingAsrManager`
- Swift 6 actor-safe with Sendable types

#### Diarizer: Embedding Skip Strategy (#480)
New `EmbeddingSkipStrategy` opt-in for `OfflineDiarizerConfig`. Skips embedding model calls when
consecutive segmentation windows have highly similar speaker masks (cosine similarity >= threshold).

- **Recommended threshold**: 0.95 (benchmarked across VoxConverse, SCOTUS, Earnings-21)
- **Speedup at higher overlap** (stepRatio=0.15): up to **2.29x** on 74-min SCOTUS audio
- **Zero quality loss**: identical DER across all test corpora
- **Default**: `.none` — opt-in only, backward compatible

---

### Bug Fixes

#### iOS: Kokoro Compute Units for iOS 26 ANE Regression (#482)
iOS 26 beta ANE compiler regression causes Kokoro models to fail with
`Cannot retrieve vector from IRValue format int32`. Fix: `KokoroTtsManager(computeUnits: .cpuAndGPU)`
bypasses the ANE on iOS 26+. Default remains `.all` for iOS 17-18.

**Our implementation already uses `.cpuOnly`** (different fix for the BNNS warm-up crash on device),
which also sidesteps this regression. No code change required.

#### TTS: Kokoro Audio Trimming Fixed (#447)
All platforms now use v1 models (was: macOS used v2 fp16). v2 models had broken `audio_length_samples`
output (always returns 0), causing 5-second silent padding. Fix: compute audio length from `pred_dur`
output (sum(pred_dur) * 600 samples/frame). "Hello world" now correctly generates 1.5s audio.

#### ASR: Use-After-Free Crash on Concurrent Transcription (#473)
`resetDecoderState()` was resetting **both** mic and system decoder states. When mic+system transcribed
concurrently, whichever finished first freed the other's in-flight `MLMultiArray` objects, causing
`EXC_BAD_ACCESS` in the autorelease pool. Fix: `resetDecoderState(for: source)` resets only the
completed source's state. Critical for meeting recorder scenarios.

#### ASR: EOU 320ms Frame Count (#444)
`StreamingEouAsrManager` with 320ms chunks produced 63 frames instead of 64, causing shape
mismatches. Fix: updated mel spectrogram formula to account for nFFT/2 center padding, matching
NeMo's computation. Our streaming ASR uses 320ms chunks as its first preference — this fix is
directly applicable.

#### ASR: Cancellation No Longer Triggers Error Recovery (#481)
`SlidingWindowAsrManager.processWindow()` previously triggered decoder reset and model re-download
when the task was intentionally cancelled. Fixed by guarding catch sites against `CancellationError`.

---

### Architecture Changes

#### Breaking API: `AsrManager.initialize(models:)` → `loadModels(_:)` (#468)
**Impact on our code**: `FluidAudioASR.swift` line 158 must change from
`try await mgr.initialize(models: models)` to `try await mgr.loadModels(models)`.

#### ASR Architecture Cleanup and Renames (#468, #440)
- `StreamingAsrManager` protocol → `SlidingWindowAsrManager` (no impact — we use concrete types)
- `EouStreamingAsrManager` → `StreamingEouAsrManager` (already updated in our code)
- `AsrManager.resetState()` → `reset()` (we do not call this directly)
- Directory reorganized into `Parakeet/` and `Qwen3/` families

#### Dependency: swift-transformers Removed (#449)
Replaced with a 145-line minimal BPE tokenizer. Eliminates dependency conflict with WhisperKit.
No functional change to our usage.

---

### Upgrade Risk Assessment

| Area | Risk | Notes |
|------|------|-------|
| `AsrManager.loadModels(_:)` API rename | **High** — build error | Must update `FluidAudioASR.swift` |
| Kokoro audio trimming (v1 models) | Low | Audio quality improvement; v1 models our Kokoro already used |
| Concurrent transcription crash fix | Low | We don't use concurrent mic+system |
| EOU 320ms frame count fix | Low-Medium | We try 320ms first; this is a correctness fix |
| CTC JA/ZH-CN new managers | None | Additive; opt-in by model ID |
| PocketTTS session API | None | Additive; new API alongside existing `synthesize()` |
| Diarizer skip strategy | None | Additive opt-in |
| swift-transformers removal | None | Package dependency change only |

---

### New Speakers/ASR/VAD Needed?

**Do not create new dedicated Speaker/ASR/VAD classes.** The new models integrate into existing classes:

1. **Japanese ASR** → `FluidAudioASR` (add `fa-parakeet-ctc-ja` model ID via `CtcJaManager`)
2. **Mandarin ASR** → `FluidAudioASR` (add `fa-parakeet-ctc-zh-cn` model ID via `CtcZhCnManager`)
3. **PocketTTS sessions** → `FluidAudioPocketTTSSpeaker` (optional streaming upgrade)
4. **Diarizer skip strategy** → `FluidAudioDiarizer` (performance opt-in for offline mode)

No new speaker, ASR, VAD, or diarizer classes are warranted by this upgrade.
