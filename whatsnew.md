# FluidAudio What's New

## Upgrade: tag-20260406 to tag-20260412

### New Features

#### ASR: Parallelized Batch Transcription, 2.2-2.8x Faster (#507)
`AsrManager` now distributes independent audio chunks across a worker pool (default: 4 concurrent
workers) for long-file transcription. Benchmarked on Apple M3 and iPhone SE 3:

| Model | Before | After | Speedup | Memory Delta |
|---|---|---|---|---|
| Parakeet TDT v2 | 31.84 s | 11.25 s | 2.83x | +21.4 MiB |
| Parakeet TDT v3 | 31.37 s | 12.75 s | 2.46x | +31.0 MiB |
| Parakeet TDT-CTC 110M | 19.89 s | 9.08 s | 2.19x | +19.7 MiB |

Concurrency is controlled by `ASRConfig.parallelChunkConcurrency` (default `4`). Transcript content
and word timings are bit-identical to the serial path. This only affects batch (file) transcription;
the streaming path is unchanged.

**iOS note:** Peak RAM increases ~20-30 MiB per transcription session. The benchmark was run on
iPhone SE 3, which has 4 GB RAM, so this is considered acceptable on modern iPhones.

#### ASR: Nemotron Streaming with 160ms and 80ms Chunk Sizes (#490)
Two finer-grained chunk size variants are now exposed in the public API via new `Repo` enum cases
`.nemotronStreaming160` and `.nemotronStreaming80`:

| Chunk | Latency | Use Case |
|---|---|---|
| 1120ms | 1.12s | Best accuracy (original) |
| 560ms | 0.56s | Lower latency |
| 160ms | 0.16s | Very low latency |
| 80ms | 0.08s | Ultra-low latency |

#### Diarizer: Custom Segment Activity Reporting (#493)
`DiarizerTimeline` now supports a `DiarizerActivityType` enum with two modes:
- `.sigmoids` (default): segment activity reported as mean probability score
- `.logits`: segment activity reported as mean logit value (covariance-compatible)

Useful for applications that need raw logits for downstream analytics.

---

### Bug Fixes

#### ASR: Japanese CTC Model Fails to Load After Download (#516)
`AsrModels.load()` and `AsrModels.download()` previously accepted CTC-only model versions
(`.ctcJa`, `.ctcZhCn`) and downloaded correctly, but then failed at load time with a cryptic
"Model file not found: Decoder.mlmodelc" error (TDT decoder name, not CTC). Both methods now
reject these versions immediately with a clear error: "CTC-only model .ctcJa must be loaded via
CtcJaManager, not AsrModels."

---

### Refactors and Breaking Changes

#### BREAKING: `Repo.parakeetCtcJa` Renamed to `Repo.parakeetJa` (#520)
The HuggingFace repository `FluidInference/parakeet-ctc-0.6b-ja-coreml` contains both CTC and TDT
v2 models, so the name `parakeetCtcJa` was misleading. Renamed to `parakeetJa`. The separate
`parakeetTdtJa` case (which previously pointed to a non-existent separate TDT repo) was removed;
both Japanese CTC and TDT models now resolve to `parakeetJa`.

**Action required:** Any call site using `Repo.overrideFolderNames[.parakeetCtcJa]` must change
`.parakeetCtcJa` to `.parakeetJa`.

#### Standardized Model Loading API Across All ASR Managers (#506)
All ASR managers now share a consistent API surface:

```swift
manager.loadModels(from: URL)                      // Load from local directory
manager.loadModels(_ models: PreloadedModels)       // Use pre-loaded models
manager.loadModels(to: URL?, progressHandler:)      // Download then load
```

| Manager | Old | New |
|---|---|---|
| `AsrManager` | `configure(models:)` | `loadModels(_:)` (deprecated old name) |
| `SlidingWindowAsrManager` | `start()` | `startStreaming()` with model loading separated |
| `StreamingEouAsrManager` | `loadModelsFromHuggingFace()` | `loadModels(from:)` |
| `StreamingNemotronAsrManager` | `loadModels(modelDir:)` | `loadModels(from:)` |

Our code already uses `mgr.loadModels(models)` (introduced at tag-20260406). No change needed.

#### File Reorganization: Batch Managers Moved into SlidingWindow/ (#502)
24 source files and 10 test files reorganized under `SlidingWindow/`:
```
SlidingWindow/
├── SlidingWindowAsrManager.swift   (public API)
├── TDT/                            (AsrManager, TdtJaManager, Decoder/)
└── CTC/                            (CtcJaManager, CtcZhCnManager)
```
Internal reorganization only. Public API and our calling code unaffected.

#### Language Model Files Deduplicated (#492)
`CtcJaModels`, `CtcZhCnModels`, `TdtJaModels` refactored from ~250 lines each into a shared
`ParakeetLanguageModels<Config>` generic with a `ParakeetLanguageModelConfig` protocol. Each
language file is now ~22 lines (config + typealias). No API change for callers.

#### `AudioConverter` Now `Sendable` (#505)
Swift 6 concurrency conformance. No behavior change.

---

### Documentation

- Diarization pipeline version distinction clarified: online/streaming uses Pyannote 3.1;
  offline batch uses Pyannote Community-1.
- ASR API reference completed and reorganized under `Documentation/`.

---

### Upgrade Risk Assessment for Privacy AI

| Change | Risk | Action |
|---|---|---|
| `Repo.parakeetCtcJa` → `Repo.parakeetJa` | **High** — build error without fix | Update `FluidAudioASR.swift` line 186 |
| Parallel batch transcription (+20-30 MiB RAM) | Low | Automatic improvement, no code change |
| Standardized loading API | None | Our code already uses the new form |
| File reorganization | None | Internal; public API stable |
| Language model deduplication | None | No caller-visible change |
| Nemotron 160ms/80ms chunk sizes | None | New option; not yet integrated |
| Diarizer activity type | None | Advanced opt-in feature |

### New Speakers/ASR/VAD/Diarizer Classes Needed?

No new classes are required for this upgrade. The existing classes cover all new functionality:
- Nemotron 160ms/80ms → can be added to `FluidAudioASR` if Nemotron streaming is integrated
- Diarizer activity type → opt-in via `DiarizerActivityType` on the existing `FluidAudioDiarizer`
