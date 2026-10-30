# FluidAudio Upgrade Notes: tag-20261024 → tag-20261030

**Merged:** 2026-10-30 · **Upstream:** FluidInference/FluidAudio `main` (**21 commits**, `925f32f4..58db32f0`),
podspec **v0.17.7 → v0.18.0** · **Merge:** `58db32f0` (conflict-free) · **Fork patch on top:** none
**Integration doc:** `helper/docs/audio-fluidaudio.md` · **Consumers:** `libs/audio/fluidaudio/`,
`libs/audio/voiceclone/fluidaudio/VoiceCloneFluidPocketBuilder.swift`, `libs/audio/ITNHelper.swift`,
`SpeechHelper.createFluidAudioSpeaker` / `makeJobSpeaker`, `views/audio/fluidaudio/FluidAudioModelManager.swift`

Most of the 21 commits add backends the app does not wrap: LFM3-Audio-2B (ASR, live ASR and TTS on Core ML
and Core AI), Phonon-2.5 multilingual GPU ASR, Underdog Bark 3.0 / 3.0 Clean, Nemotron Unified ASR 0.6B,
Nemotron 3.5 Transcribe Lite and Paradee-8M v1.2. The rest is vocabulary-boosting and sliding-window
streaming work (#1041, #1080, #929, #1079, #1035), which the app does not call either. Only **three commits
reach a shipped row**, and all three are small: Parakeet transcripts drop a stray `<unk>` (#1054), the EOU
streaming manager now throws `CancellationError` from a cancelled task (#1033), and the PocketTTS tokenizer
no longer matches literal special-piece text such as `<s>` (#1068). **No silent ship-blocker**: no fork patch
was lost or moved (49 markers / 13 files, P3), no `ModelNames` constant of a shipped row changed, no new
network path is reachable from a shipped variant, and no app code needs an edit to compile.

---

## 1. Upstream changes, by area

### 1.1 ASR, the shipped paths (`AsrManager` batch for v3 / ja, `StreamingEouAsrManager` for EOU)

| Commit | What it does | Reaches the app? |
|---|---|---|
| `6dac59a7` #1054 | Adds `.bark3` / `.bark3Clean`. Also: `<unk>` tokens are dropped before text and timings are built (`droppingUnknownTokens`, plus a filter in `AsrManager`'s token join) for **every** TDT version | **Yes**, `fa-parakeet-tdt-v3` and `fa-parakeet-ctc-ja`: a literal `<unk>` can no longer appear in a transcript. Text otherwise unchanged |
| `6924debc` #1033 | `SlidingWindowAsrManager.cancel()` joins its worker. **EOU** `process(audioBuffer:)`, `finish()` and `processBufferedAudio()` now `try Task.checkCancellation()`; the Nemotron streaming managers too | **Yes**, `fa-parakeet-eou-120m`: a cancelled feed/finish task now throws `CancellationError` instead of running to the end. Stop paths audited in §5.1: no user-visible effect |
| `2d0ef4bc` #1075 | `AsrModelVersion.phonon25`; `resolvedEncoderComputeUnits` only changes the placement for `.phonon25`; `inferredVersion` exact-folder match; pinned-revision download short-circuit | No: for `.v3` / `.tdtJa` the encoder units are still `requested ?? configuration`; the app passes the version explicitly and loads with `loadLocal` |
| `4d1b57bc` #1035 | `SlidingWindowTokenUpdate` / `SlidingWindowTokenCursor` token-revision stream; a monotonic stable-prefix floor in the re-decode (`stableTokenFloor`, default 0) | No: the app does not use `SlidingWindowAsrManager`; with floor 0 the shared re-decode helper is unchanged |
| `1aa042c7` #1041 | Vocabulary rescoring keeps names, possessives and the word after a hyphenated name | No: the app configures no custom vocabulary |
| `2cbae2b5` #1080 | `VocabularyBoostingSession.Thresholds`; `detectedTerms` floor −6.5; StreamingUnified double-space fix | No (no vocabulary boosting, no Unified manager) |
| `2faffc23` #929 | `ASRResult.ctcReplacements` (defaulted, `nil` without boosting); `TextNormalizer` forwards it | No behaviour change; ITN path unchanged |
| `6f793893` #1079 | `CtcModels` MelSpectrogram forced to `.cpuOnly` (macOS 27.0.1 put part of it on the ANE and the CTC encoder emitted blank) | No: `CtcModels` serves keyword spotting / vocabulary boosting only; `fa-parakeet-ctc-ja` runs on `AsrManager(.tdtJa)`, which never loads `CtcModels` |
| `55f65d9c` #1071 | Nemotron Unified ASR 0.6B (25 European languages) on the Unified managers | Not adopted (§6) |
| `c9cad1bb` #1069, `6ea419dc` #1070 | Nemotron 3.5 Transcribe Lite (encoder + CTC words, speakers from Nemotron-3-Diarization); OS check before download. `AudioMelSpectrogram` gained optional `window:` / `filterbank:` | Not adopted (§6). The mel change is defaulted (`nil`), so EOU and `Nemotron3Diarizer` compute the same features |
| `4e2985fc` #1076, `6b516c3d` #1078, `fc5c5f06` #1077 | LFM3-Audio-2B (Core ML + Core AI), low-disk hardening, `ModelCachePreservingError` in `ModelHub.loadWithRecovery`, TTS host-work trims | Not adopted (§6). The `ModelHub` change only skips a purge-and-redownload for that error type |

### 1.2 TTS

| Commit | What it does | Reaches the app? |
|---|---|---|
| `4c65ad49` #1068 | `SentencePieceTokenizer` matches only normal / user-defined pieces; an uncovered scalar becomes byte pieces (byte fallback on) or one `<unk>` per run (off) | **Yes, edge case only**, all 6 `fa-pocket-tts-*` rows: every shipped `tokenizer.model` has byte fallback on (upstream checked all 28), so ordinary text tokenizes as before. Input that literally contains `<s>`, `<unk>`, `<0x41>` now tokenizes as text instead of the control piece. Upstream: 184/184 token-ID matches with Google `sentencepiece` (was 145/184) |
| `a7c887af` #1073 | Paradee-8M v1.2 (four voices, `.v12` default), British G2P: `gb_lexicon_cache.json`, `MisakiBritishFallback`, `KokoroAneEnglishPhonemizer(british:)` (default `false`), `LexiconAssetCache.englishFileName(british:)`, `ensureEnglishLexicon(british:)` | Not adopted (§6). Kokoro ANE stays on the US lexicon (`british: false`); the fork's flat-lookup gate keys on the chosen filename (P3) |
| `1b0ed50d` #1052 | Pocket small demo app fixes | No (Examples) |

### 1.3 Packaging and other

- `e1528798` podspec 0.18.0. No `Package.swift` / `Package@swift-6.2.swift` change in this range.
- `09359e2c`, `55d9dab7`, `3f6e600d`: showcase docs. Nothing.

---

## 2. What users get from this merge

- **Parakeet transcripts never show a stray `<unk>`** (#1054). Rare on v3 and Japanese, but it could appear
  where the audio held a character the vocabulary lacks.
- **Live dictation stops cleanly when cancelled** (#1033): a cancelled EOU session stops decoding at once
  instead of draining queued audio first.
- **PocketTTS reads literal `<s>` or `<unk>` in the text as text** instead of a control token (#1068).

No new models, languages or voices reach users in this merge; they need app code (§6).

## 3. Risk assessment — 5 risks (1 high, 0 medium, 4 low)

### R1 — iOS 27 Kokoro advisory still in force; macOS cannot verify it (HIGH, unchanged)
**Problem:** `KokoroAneManager.swift` is byte-identical to tag-20261024 (P3), so `isBnnsCrashProneOS` is still
**false on macOS 27** and true on every non-macOS OS ≥ 26.4. A green macOS row says nothing about iOS 27 (#889:
libBNNS `vadd_fp16_sme_internal` on the Metal-free route, MPSGraph abort on the Metal route). Nothing in this
range touches it. **Verify:** physical iOS 27 device, long English Text to Audio export. **Status:** needs device.

### R2 — EOU `CancellationError` on cancelled tasks (LOW)
**Problem:** #1033 makes `process(audioBuffer:)` and `finish()` throw when their task is cancelled. In
`FluidAudioASR.finishStreaming()` a throw skips closing the stream and leaves `isStreaming` /
`isTranscribingPublished` set. Every caller that cancels the task also calls `cancelStreaming()`, which resets
both, and the normal Stop path never cancels before `finish()` (§5.1). **Verify:** P10
`run_fluidaudio_tests.py` EOU row; on device, cancel keyboard dictation mid-utterance, then dictate again.
**Status:** audited, no fix needed.

### R3 — Parakeet `<unk>` filtering (LOW)
**Problem:** #1054 changes v3 / ja text only where `<unk>` was emitted. Timings stay aligned (the parallel arrays are
filtered together). **Verify:** P10 `fa-parakeet-tdt-v3` and `fa-parakeet-ctc-ja` vs the P1 transcripts.
**Status:** pending P10.

### R4 — PocketTTS tokenizer rewrite (LOW)
**Problem:** #1068 rewrites the matcher (piece types, user-defined scoring, unknown penalty from the minimum
normal-piece score). Upstream states ordinary text is token-identical for every shipped pack; the regression
probes have no oracle (`[no oracle]` in P1). **Verify:** P10 6 PocketTTS rows (sample count / RMS close to P1).
**Status:** pending P10.

### R5 — Trait drift (LOW, unchanged)
**Problem:** a future upstream default-trait change could drop TTS/Diarizer silently. Not moved in this range.
**Verify:** the app build compiles `FluidAudioKokoroAneSpeaker` and `FluidAudioDiarizer`. **Status:** P10 build.

No `ModelNames` constant used by a live row changed and no catalogue `components` change is needed (P3).

---

## 4. Fork patches

| File | Resolution |
|---|---|
| `ASR/Parakeet/Streaming/EOU/StreamingEouAsrManager.swift` | Upstream added 3 `try Task.checkCancellation()` lines (#1033); the fork's per-utterance EOU reset (2 markers) is intact |
| `ModelNames.swift` | Additive upstream entries only; `Repo.overrideFolderNames` and the `folderName` override check intact |
| `TTS/KokoroAne/Assets/KokoroAneResourceDownloader.swift` | `ensureEnglishLexicon(british:)` picks `gb_` / `us_lexicon_cache.json`; the fork's flat-lookup-then-nil gate uses that filename, so it holds for both (only Paradee passes `british: true`). 14 markers intact |

**Inventory:** 13 files, **49 → 49** markers (`git grep -c "wangqi modified"` at `925f32f4` and the merged tree
match file by file). The other 10 patched files and both manifests are byte-identical to `925f32f4`.
Nothing absorbed, nothing added.

### Dependencies

No dependency move. Both manifests keep the fork's Swift 5 language mode; TTS/Diarizer traits default-on.

---

## 5. `libs/audio/fluidaudio/` audit

What the app calls, checked against the merged sources. **No consumer fix**: every call compiles unchanged and no
behaviour change needs app code. No app code switches exhaustively over `AsrModelVersion` (grew `.phonon25`,
`.bark3`, `.bark3Clean`) or `ParadeeVariant`. `FluidAudioModelManager` does not import FluidAudio.

| Symbol the app calls | Change | Our usage / fix |
|---|---|---|
| `AsrModels.loadLocal(from:version:)`, `AsrManager.transcribe(_:decoderState:)` | `<unk>` dropped from text / timings (#1054); placement unchanged for `.v3` / `.tdtJa` | none |
| `StreamingEouAsrManager(configuration:chunkSize:)`, `loadModels(from:)`, `reset()`, `process(audioBuffer:)`, `finish()`, callbacks | `process` / `finish` throw `CancellationError` on a cancelled task (#1033) | none (§5.1) |
| `CoherePipeline`, `ParaformerModels`, `SenseVoiceModels`, `VadManager`, `Nemotron3Models` / `Nemotron3Diarizer` | Unchanged (`AudioMelSpectrogram` gained defaulted parameters only) | none |
| `KokoroAneManager`, `KokoroAneConstants`, `TtsCacheDirectory.overrideDirectory` | Unchanged; English phonemizer `british:` defaults `false` | none |
| `PocketTtsManager(language:directory:skipDownload:)`, `synthesize(...)`, `PocketTtsResourceDownloader.ensureMimiEncoder` | Tokenizer edge case (#1068) | none |
| `ASRResult` (FluidAudio) / `TextNormalizer.shared` (ITNHelper) | `ctcReplacements` added, defaulted `nil` | none |

### 5.1 EOU stop paths and #1033

`FluidAudioASR.feedAudio` → `mgr.process`, `finishStreaming` → `mgr.finish()`. Callers:
- **Keyboard** (`apps/keyboard/KeyboardSessionHandler.swift`): Stop finishes the feed continuation; the consumer
  drains and calls `finishStreaming` uncancelled, so `.final` arrives as before. Cancel / expiry cancels the
  consumer, then calls `cancelStreaming()`; `finishStreaming` is under `try?`, so the error is swallowed and
  state resets. A feed in flight at that moment now logs `streaming feed error: CancellationError` at
  **warning** level (new, cosmetic).
- **Werewolf dictation**: Stop finishes the feed; the feed task is never cancelled (the 3 s wait cancels only
  its own group children). `finishStreaming` is under `try?`. No effect.
- **ASR test view** (`views/audio/ASRPlayTestView.swift`): Stop finishes the feed, uncancelled. Cancel
  (`onDisappear`, `cleanupStreamingEngine`) cancels first and checks `Task.isCancelled` before finishing; only a
  cancel landing *inside* `finish()` reaches its `transcribeError`, and that happens only from `onDisappear`,
  when the view is gone. The `cleanupStreamingEngine` callers are start-failure paths, before any finish.

No error reaches the user, so per the brief no fix was made. Optional hardening (not built): in
`FluidAudioASR.finishStreaming`, catch `CancellationError` and close the stream like `cancelStreaming()`, and
in `feedAudio` rethrow it without the callers' warning.

---

## 6. New models that would need new app code

Sizes are HuggingFace tree sums (2026-10-30); "load set" is the compiled subset the loader opens.

| Model | Size | New class? | Upstream API it would wrap |
|---|---|---|---|
| **Gradium Phonon 31m** (#1053, English, Apache-2.0, iOS 18 / macOS 15) | `pocket-tts-coreml/phonon/phonon-31m-en` 667 MB tree; **load set 126 files, 173 MB** (`pocket_state.mlmodelc` 80 MB, `mimi_decoder.mlmodelc` 45 MB fp32, `arch.json`, `pocket_state.json`, `constants_bin/`, `voices/` 26 MB) | **Yes: `FluidAudioPocketTTSSmallSpeaker`** (+ SpeakerType case + catalogue row). Reported, not built | `PocketTtsSmallManager(model: .phonon31m)`, **`load(from:)`** (local, no network) + `synthesize` / `makeSession` streaming, `availableVoices()`. 4 KV-snapshot voices in `constants_bin/` (`default`, `en_gb_Maeve`, `en_us_Easton`, `en_us_Zoey`) + ~120 conditioning voices in `voices/*.safetensors`. No cloning. `initialize()` downloads via `ModelHub` with **no offline gate** (the fork's `skipDownload` covers only `PocketTtsManager`), so never call it |
| **LFM3-Audio-2B** (#1076, ASR + live ASR + TTS) | `lfm3-audio-2b-coreml` 4.8 GB (`coreml/` 2.3 GB, `ane/` 1.8 GB, root assets 0.7 GB; the Core ML pack ~3.3 GB); `lfm3-audio-2b-coreai` 2.4 GB | **Yes**: an ASR class (`FluidAudioLFM3ASR`) and a speaker (`FluidAudioLFM3Speaker`), or one class for both. Over the 700 MB ASR / 1000 MB TTS caps | `LFM3AudioManager` (`transcribe`, `startLiveTranscription`, speech), `LFM3AudioModels.load`; GPU, foreground only |
| **Phonon-2.5** multilingual (#1075) | `Muesli-HQ/phonon-2.5-coreml@a042ef89` 290 MB; ~480 MB peak | **No**: a `FluidAudioASR` branch + `repoMap` + row | `AsrModels.loadLocal(from:version: .phonon25)`; encoder `.cpuAndGPU` only (ANE throws), so **not background-safe** |
| **Underdog Bark 3.0 / 3.0 Clean** (#1054, English, verbatim / no fillers) | `bark-3.0-coreml/bark-3.0` 689 MB, `bark-3.0-clean` 700 MB (whole subfolder) | **No**: same branch as Phonon-2.5 | `loadLocal(version: .bark3 / .bark3Clean)`, iOS 18+ |
| **Nemotron Unified ASR 0.6B** (#1071, 25 European languages) | repo 14.4 GB; offline int8 load set ≈ 625 MB (`encoder_int8` 600 + decoder 18 + joint 7), fp16 ≈ 1.2 GB; streaming encoders ~595 MB int8 each | **Yes**: `FluidAudioUnifiedASR` (the app has no Unified manager wrapper) | `UnifiedAsrManager` / `StreamingUnifiedAsrManager` with `UnifiedConfig` |
| **Nemotron 3.5 Transcribe Lite** (#1069) | repo 6.5 GB; Core ML load ≈ 1.36 GB (`lite_encoder_w48` 1263 + `lite_ctc_head_w48` 101); Core AI `lite_ctc_gpu.aimodel` 1.36 GB | **Yes**: `FluidAudioNemotronLiteASR`. Over the 700 MB ASR cap | `NemotronTranscribeLiteManager` (+ `NemotronTranscribeLiteSpeakers` with the shipped Nemotron-3 diarizer) |
| **Paradee-8M v1.2** (#1073, 4 voices incl. 2 British) | `v1.2/int8` 12 MB, `v1.2/fp32` 35 MB; British voices also need `gb_lexicon_cache.json` | **Yes**: `FluidAudioParadeeSpeaker` | `ParadeeManager(version: .v12, variant:directory:)`, `synthesize(text:voice:)` |

Carried from tag-20261024, still not built: Phonon-2, Parakeet Ultra / Redux (`FluidAudioASR` branch);
Canary translation (new class); PocketTTS small 18m/7m/2m (same new speaker as Phonon 31m); Kokoro ANE-v3;
Supertonic-3 v2.

Opportunities (not built): Phonon-2.5 as a faster multilingual Parakeet for foreground transcription; Bark 3.0
Clean for Meetings transcripts without fillers; the Nemotron Unified int8 encoder as a 25-language ASR within the
cap; Paradee v1.2 as a 12 MB English voice for low-memory devices; `ctcReplacements` if custom vocabulary is ever
added.

---

## 7. Deliberately not adopted

| Upstream addition | Why not |
|---|---|
| Phonon 31m, Paradee v1.2, LFM3-Audio, Nemotron Unified, Nemotron 3.5 Lite, Phonon-2.5, Bark 3.0 | Need new app code / catalogue rows (§6); out of scope for an engine upgrade |
| Vocabulary boosting thresholds, rescoring fixes, `ctcReplacements`, CTC mel on CPU | The app configures no custom vocabulary |
| Sliding-window token-revision stream (#1035) | The app streams with EOU, not `SlidingWindowAsrManager` |

---

## 8. Verification

| Check | Result |
|---|---|
| P1 baseline (`run_fluidaudio_tests.py`) | 18 passed, 1 failed, 0 skipped; standing FAIL `fa-silero-vad` (permission error saving into the staged folder, doc OPEN) |
| P3 patch audit | 49 / 13 files intact; no shipped `ModelNames` change; no new network path on a shipped variant |
| `swift build` (fork, P3) | Skipped (cold build during a running sweep); covered by the app build |
| App builds, `FluidAudioCatalogueSchemaTests`, `FluidAudioASRDispatchTests`, `run_fluidaudio_tests.py --compare <P1>` | Pending P10 |

**macOS cannot verify:** the iOS 27 Kokoro advisory (R1: `isBnnsCrashProneOS` is false on macOS 27); keyboard
dictation cancel on device (R2); background (ANE-only) behaviour of every row.
