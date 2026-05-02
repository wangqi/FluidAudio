# FluidAudio: tag-20260425 → tag-20260502 Upgrade Notes

## Commits Included

| # | Commit | Title |
|---|--------|-------|
| 1 | 7c115f6 | feat(tts/kokoro-ane): add laishere 7-stage CoreML chain (ANE-optimized) |
| 2 | 982f117 | fix: avoid misleading confidence warning in SlidingWindowAsrManager.finish() |
| 3 | eff1752 | feat(tts/pocket): multi-language support (EN + 9 new packs) |
| 4 | b82d4f2 | feat(tts): CosyVoice3 Mandarin zero-shot TTS port |
| 5 | 3d9d422 | feat(tts/magpie): add NVIDIA Magpie TTS Multilingual 357M Swift port |
| 6 | d89cf01 | docs(models): list CosyVoice3 under Not Production Ready |
| 7 | e435319 | docs(models): drop Parakeet CTC Japanese + ASR/TTS row cleanups |
| 8 | 5c16ee1 | docs(models): add Cohere Transcribe + Qwen3-ASR rows |
| 9 | e332c18 | docs(models): fix Cohere Transcribe Model Sources link target |
| 10 | 248b76b | feat(tts/styletts2): scaffold StyleTTS2 4-stage pipeline integration |
| 11 | 00ea906 | fix: remove module_map from MachTaskSelfWrapper subspec |
| 12 | c4d56a5 | Feat/pocket tts int8 precision swap |
| 13 | 3e3ee69 | docs: add top-level architecture overview |
| 14 | 4db4af1 | Add Dictato to showcase |
| 15 | 4065a99 | Optimized LS-EEND API (major refactor) |
| 16 | 35f6ba6 | Added Back the Old LS-EEND Constructors |
| 17 | b5d8017 | feat(asr/parakeet-v3): default to int4-per-channel encoder |
| 18 | 7603ac6 | feat(tts/benchmark): tts-benchmark CLI covering all TTS backends |
| 19 | cad8a2b | feat(asr/cohere): long-form transcribeLong + cold/warm docs |
| 20 | 5bb84bc | Fix DiarizerTimeline Short Segment Filter |
| 21 | 0a9aace | Fixed short segment filter for trailing tentative segments in DiarizerTimeline |

---

## Feature Summary

### TTS Improvements

#### Kokoro ANE — 7-Stage CoreML Pipeline (#547)
A new ANE-optimized Kokoro variant uses a 7-graph CoreML chain. Benchmark on M2:
- TTFT p50: 1586 ms (vs 3113 ms for CPU+GPU Kokoro)
- RTFx: 5.19x real-time
- WER: 10.8% on MiniMax English corpus

Not integrated in this release. Our `FluidAudioKokoroSpeaker` uses the standard Kokoro variant which is simpler to initialize and production-ready. A separate model download (`laishere/kokoro-82m-coreml-ane-optimized`) would be needed.

#### PocketTTS Multi-Language Support (#549)
PocketTTS now supports 10 language packs: `english`, `french`, `german`, `spanish`, `italian`, `portuguese`, `chinese`, `japanese`, `korean`, `arabic`.

API change: `PocketTtsManager(directory:)` gains a `language: PocketTtsLanguage` parameter that defaults to `.english`. Backward compatible — our code is unchanged. Language switching requires creating a new manager instance (language is immutable for the lifetime of one manager).

#### PocketTTS Int8 Precision Swap (#558)
New `PocketTtsPrecision { .fp16, .int8 }` enum for `PocketTtsManager`. The int8 `flowlm_stepv2.mlmodelc` variant reduces memory. Default stays `.fp16` — no behavior change.

#### BETA: NVIDIA Magpie TTS Multilingual 357M (#541)
New `MagpieTtsManager` for multilingual TTS at 22.05 kHz. Status: BETA — RTFx 0.64x (below real-time), TTFT p50 ~9.6 s. Emits a runtime warning on `initialize()`. Not suitable for production use.

#### BETA: CosyVoice3 Mandarin Zero-Shot TTS (#536)
New `CosyVoice3TtsManager` for Mandarin/Cantonese zero-shot voice cloning. Status: BETA — RTFx 0.25-0.36x, requires chunking for long phrases. Emits a runtime warning. Not suitable for production use.

#### BETA: StyleTTS2 Integration (#554)
New 4-stage StyleTTS2 pipeline scaffold. Status: BETA — WER ~44%, RTFx 2.72x but only on ~22 s/phrase output. Emits a runtime warning. Not suitable for production use.

---

### ASR Improvements

#### Cohere Transcribe: Long-Form Audio Support (#564)
New `CoherePipeline.transcribeLong(audio:models:language:)` handles audio longer than 35 seconds. The prior `transcribe()` silently truncated audio at the 35 s encoder window. `transcribeLong` slices into 35 s chunks with 5 s overlap and stitches chunks via token-level longest-common-substring merge.

`transcribeLong` is a full superset: audio <= 35 s short-circuits to `transcribe()` with byte-identical output. Our `FluidAudioASR.swift` has been updated to always call `transcribeLong` for the Cohere backend.

#### Parakeet v3: Int4 Encoder Default (#560)
Parakeet TDT v3 encoder now defaults to `int4-per-channel` quantization (was int8). Reduces model size and may improve throughput on Apple Neural Engine. No API change needed — the encoder precision is baked into `AsrModels.load(from:version:)`.

---

### Diarization Improvements

#### LS-EEND API Refactor (#526) — Breaking Change, Fixed
Major restructuring of the LS-EEND diarizer. `LSEENDModelDescriptor` and `initialize(descriptor:)` have been removed. Metadata is now embedded inside the `.mlmodelc` model description, so no separate JSON file is needed.

Old API (broken after this upgrade):
```swift
let descriptor = LSEENDModelDescriptor(variant: .dihard3, modelURL:..., metadataURL:...)
let diarizer = LSEENDDiarizer()
try diarizer.initialize(descriptor: descriptor)
```

New API:
```swift
let model = try LSEENDModel(modelURL: modelURL.appendingPathComponent(modelFile))
let diarizer = LSEENDDiarizer()
try diarizer.initialize(model: model)
```

Our `FluidAudioDiarizer.swift` has been updated. The `LSEENDDiarizer()` no-arg constructor (no throws) is available again via the restored constructors in #563.

#### DiarizerTimeline Short Segment Filter Fix (#565, #566)
Two-part fix:
1. Gaps were closed as soon as any speech appeared instead of waiting for a second segment of sufficient length — `config.minFramesOn` threshold is now correctly applied.
2. Trailing tentative segments at the end of a buffer emitted incorrect merged spans and wrong activity scores.

These fix streaming diarization segment boundary accuracy. No API change.

---

## Breaking API Changes Requiring Code Updates

| Area | Breaking Change | Our Fix |
|------|----------------|---------|
| LS-EEND init | `LSEENDModelDescriptor` removed; `initialize(descriptor:)` removed | Use `LSEENDModel(modelURL:)` + `initialize(model:)` in `FluidAudioDiarizer.swift` |
| Cohere ASR | `transcribe()` silently truncated audio > 35 s | Switch to `transcribeLong()` in `FluidAudioASR.swift` |

---

## Risk Assessment

| Change | Risk | Notes |
|--------|------|-------|
| LS-EEND API refactor | Medium | Breaking API fixed; runtime behavior unchanged since we load same local files |
| Cohere transcribeLong | Low | Superset of old transcribe(); short audio byte-identical, long audio now works correctly |
| PocketTTS multi-lang | Low | Default .english unchanged; no behavior change |
| PocketTTS int8 | Low | Default stays .fp16; opt-in only |
| Parakeet v3 int4 default | Low | Smaller model, faster inference; quality equivalent or better per upstream testing |
| DiarizerTimeline fixes | Low | Bug fixes only; segment boundaries more accurate |
| BETA TTS (Magpie/CosyVoice3/StyleTTS2) | None | Not integrated; upstream marks as BETA with runtime warning |
| Kokoro ANE variant | None | Not integrated; requires separate model download |

---

## New Models Available (not yet integrated)

| Model | Type | Status | Notes |
|-------|------|--------|-------|
| NVIDIA Magpie TTS 357M | TTS (multilingual) | BETA | Below real-time |
| CosyVoice3 | TTS (Mandarin/Cantonese zero-shot) | BETA | Below real-time |
| StyleTTS2 | TTS (multi-speaker) | BETA | WER ~44% |
| Kokoro ANE | TTS | Production | ANE-optimized; requires separate `laishere/kokoro-82m-coreml-ane-optimized` download |
| PocketTTS fr/de/es/it/pt/zh/ja/ko/ar | TTS | Production | 9 new language packs; requires `PocketTtsLanguage` on init |

Recommendation: Kokoro ANE and PocketTTS language packs are the most viable additions for the next release. BETA backends should wait for upstream quality improvements.
