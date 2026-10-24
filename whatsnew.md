# FluidAudio Upgrade Notes: tag-20260928 → tag-20261024

**Merged:** 2026-10-24 · **Upstream:** FluidInference/FluidAudio `main` (**53 commits, 2026-09-29 → 2026-10-21**),
podspec **v0.17.4 → v0.17.7** · **Merge:** `97c26284` · **Fork patch on top:** `59d2b4ae` (Kokoro ANE-zh offline gate)
**Integration doc:** `helper/docs/audio-fluidaudio.md` · **Consumers:** `libs/audio/fluidaudio/`,
`libs/audio/voiceclone/fluidaudio/VoiceCloneFluidPocketBuilder.swift`, `libs/audio/ITNHelper.swift`,
`SpeechHelper.createFluidAudioSpeaker` / `makeJobSpeaker`, `views/audio/fluidaudio/FluidAudioModelManager.swift`

389 files changed upstream (+34,854 / −2,001); 254 under `Sources/` (+9,712 / −1,368). Most of the added
lines are new backends the app does not wrap: Kokoro ANE-v3 (opt-in runtime, native Japanese), Canary
speech translation, Phonon-2 ASR, small PocketTTS models (18m/7m/2m) and Gradium Phonon 31m, Paradee-8M,
Supertonic-3 v2, diarizer embedding APIs, tvOS/visionOS builds, and the TTS/Diarizer package traits.

The most important item for our integration was a **silent ship-blocker in #1025** (`3bedd976`): upstream
started passing `directory: store.directory` to the Kokoro lexicon and multilingual G2P loaders, so the fork's
`directory == nil` offline gates stopped firing and French/Spanish Kokoro would have downloaded from
upstream HuggingFace into the app's pack. Fixed in the merge (§4). The second was that #1028 made Mandarin
g2pW actually run, which turned a dormant lazy fetch into a real network fetch on first Mandarin use; fixed by
`59d2b4ae` plus four mirror uploads (§4). No public API the app calls changed; nothing in the app needs an edit
to compile.

---

## 1. Upstream changes, by area

### 1.1 Kokoro ANE, legacy runtime (the app's `fa-kokoro-82m`, `.legacy`)

| Commit | What it does | Reaches the app? |
|---|---|---|
| `b8eab739` #999 | English phonemizer applies misaki's flap step `ɾ → T` ("metal" no longer read as "mattle"); the glottal stop is kept | **Yes**, English voices. Phonemes change for flap words |
| `feb505b4` #974 | `EnglishTextNormalizer` reads roman-numeral list markers `(i)`, `(ii)`, `ii.` as numbers, with or without NeMo | **Yes**, English voices |
| `ea63ac36` #986 | Mandarin erhua: only a real 儿 suffix folds into the previous syllable; 二/而/耳/尔 and 儿 "child" (女儿) keep `ér`. Digits too (`12` → 十二) | **Yes**, `ANE-zh` |
| `90c1b07e` #1028 | g2pW polyphone disambiguation bound to the shipped bundle's real signature; `ANE-zh/g2pw/vocab.txt` published upstream. Before, it always fell back to dictionary-only | **Yes**, `ANE-zh`, once the pack has `vocab.txt` (§4, §3 R2) |
| `3bedd976` #1025 | `G2PModel` / `MultilingualG2PModel` take a per-instance models root; Kokoro passes `store.directory` to lexicon and G2P loaders | **Yes**, broke the fork's offline gates; re-ported (§4) |
| `d8d87042` #1047 | `ModelHub` no longer sweeps every root-level `.json` of `kokoro-82m-coreml` on a `g2p-only` download (~42 MB → ~1.5 MB) | No: the app's pack is staged by `DownloadManagerCoreML` |
| `81450851` #970 | `englishPhonemes(for:)` for mixing English into other variants | No (API only) |
| `e560c644` #1009, `fc8c3e4f` #1006 | Docs: legacy runtime on A12 (iPhone XR) can load Vocoder yet fail synthesis; English dictionary vs BART | Information (§3 R5) |
| `ca0cbe77` #997, `688007bd` #1038, `bd958672` #1048, `7c422a7f` #1000 | Kokoro ANE-**v3** runtime (`version: .v3`, pinned downloads, native Japanese) and its fixes, incl. **#1038 macOS 27 all-NaN Vocoder output fixed by 16 KB tail slack in the v3 array builders** | No: the app uses `.legacy` (the legacy builders already had the slack, tag-20260928) |

### 1.2 ASR

| Commit | What it does | Reaches the app? |
|---|---|---|
| `61d41b3c` #1032 | `tdtJa`: the word-boundary splice guard is off for the unspaced Japanese vocabulary; audio > 15 s no longer drops tokens at chunk seams (`愛をつかいます。` was `愛をつ。`) | **Yes**, `fa-parakeet-ctc-ja`, long audio |
| `def9e7d1` #975 | Long-form TDT takes seam-token timing from the window that heard it mid-window; removes false gaps in word timings | **Yes**, Parakeet v3/ja segment times on audio > 15 s |
| `b6d3ebde` #1020 | `WindowProgress.windowCount` matches the windows decoded; plan laid out up front | Internal only (app does not set a window handler) |
| `793a2c5a` #1019 | Phonon-2 multifunction encoder ladder + pause-cut windows | No: v3 / ja encoders are single-function; `loadLocal` loads no ladder |
| `a8f14823` #980 | `AsrModelVersion.phonon2` (FermionResearch 5-value v3, English, iOS 18+) | Not adopted (§6) |
| `be85dfff` #833, `82eb4c1e` #1030 | Canary speech translation (en ↔ 24 European languages) and VAD pause-aligned long-form | Not adopted (§6) |
| `53e0126f` #1046, `d31145a9` #1024 | Cohere: encoder/decoder default to ANE with per-model compute units (`.all` put the INT8 encoder on GPU: 2.1 → 12.4 GB); 256-slot v3 decoder | Dormant: no live Cohere row; app pins `.v2` and passes `.all` explicitly (§3 R6) |
| `bd11ac8d` #1034, `0b0fa2ad` #993, `e05ab923` #1003 | CTC keyword scoring memory/CPU, padding fix | No: no keyword spotting in the app |
| `038b50f9` #994, `e90e84f8` #1005 | Nemotron multilingual streaming partial-callback fixes | No: not wrapped |
| `16c7dd6a` #984 | CLI `transcribe --model-version` | No |

### 1.3 PocketTTS

| Commit | What it does | Reaches the app? |
|---|---|---|
| `bcac3851` #1051 | Six small English models (18m/7m/2m + `_refl`) on a new `PocketTtsSmallManager` (FlowLM on ANE), iOS 18+ | Not adopted (§6) |
| `c79c4106` #1053 | Gradium Phonon 31m on the small host, Phonon text normalizer | Not adopted (§6) |
| (#1051) `cacheLen:` parameter threaded through `PocketTtsSynthesizer` | Defaulted to `kvCacheMaxLen` | No behaviour change for the shipped 6 language rows |

### 1.4 Diarization

`0c0f113e` #992 per-chunk embeddings from `DiarizerManager`; `018ed6b4` #1018 clean per-speaker enrollment
embeddings; `8d13acaf` #1015 `OfflineDiarizerManager.embedSpans`; `a00964e1` #1014 offline segmentation batch
size; `3e5a323b` #988 / `a5f70903` #982 offline FBank / short-span fixes; `04e363c2` #977 adopt unmarked caches
for pinned revisions. **None reaches the app**: `fa-nemotron3-diarizer` uses `Nemotron3Models` /
`Nemotron3Diarizer`, which only gained `#if Diarizer` lines. #977 removes the tag-20260928 §3.4 "pinned diarizer
wipes a staged folder" risk for future pyannote rows when the files match the listing.

### 1.5 Packaging and other

- `b37a232a` #990/#1001 **TTS and Diarizer opt-out traits** in `Package@swift-6.2.swift`. The fork keeps them
  default-on (`.default(enabledTraits: ["TTS", "Diarizer"])`); `Package.swift` (tools 6.0) defines both
  unconditionally. The app links the package as a local reference with default traits, so it still gets both.
- `224de0fb` tvOS 17+, `582d7cdb` visionOS build. Inert.
- `184c1118` Supertonic-3 v2 bundles, `4ba3e015` Paradee-8M [beta], LuxTTS/Supertonic license docs. Not wrapped.
- Showcase/README/CI commits (8). Nothing.

---

## 2. What users get from this merge

- **Japanese Parakeet keeps every word on long recordings.** Past 15 s, words between chunks were dropped up to the
  next punctuation mark (#1032).
- **Word timings on long Parakeet transcripts no longer open false gaps** at chunk seams, so Meetings segment
  times sit closer to the speech (#975).
- **Kokoro English reads flap words correctly** ("metal", "water", "better") and reads list markers `(i)`, `(ii)`
  as one, two (#999, #974).
- **Kokoro Mandarin picks the right reading for polyphonic characters** (g2pW now actually runs, #1028) and no
  longer swallows 二 / 而 / 耳 or 儿 "child" after another syllable (#986). Needs `vocab.txt` in the pack (R2).
- **Kokoro stays fully offline**, including Spanish, French and Mandarin text assets (§4); before the fork
  re-port, #1025 would have made Spanish/French download from upstream on first use.

## 3. Risk assessment — 7 risks (1 high, 2 medium, 4 low)

### R1 — iOS 27 Kokoro advisory still in force; macOS cannot verify it (HIGH, unchanged)
**Problem:** `KokoroAneManager.isBnnsCrashProneOS` is **false on macOS 27** and true on every non-macOS
OS ≥ 26.4, so a green macOS regression row says nothing about iOS 27. The iOS 27 message (#889: libBNNS
`vadd_fp16_sme_internal` on the Metal-free route, MPSGraph abort on the Metal route) is unchanged. #1038's
macOS 27 NaN fix is in the **v3** builders only; the legacy builders got the equivalent slack in tag-20260928,
and upstream has still not verified that it covers #889 on iOS. #997 states physical iPhone/iPad v3 stability is
unvalidated. **Verify:** physical iOS 27 device, long English Text to Audio export. **Status:** needs device.

### R2 — Existing ANE-zh packs lack g2pW vocab and jieba tables (MEDIUM)
**Problem:** with `59d2b4ae` the app no longer fetches them, so packs downloaded before the mirror upload run
Mandarin dictionary-only (no g2pW) and without HMM segmentation, unless an earlier lazy fetch filled
`ANE-zh/g2p/`. Not a failure. **Fix:** bump `fa-kokoro-82m` `components.revision` 3 → 4 (re-fetches only
pending files, ~400 KB). **Verify:** `run_fluidaudio_tests.py --only kokoro` `ANE-zh` row. **Status:** pending P8/P10.

### R3 — Mandarin and English phonemes change (MEDIUM)
**Problem:** #986 erhua, #1028 g2pW and polyphone catalogue, #999 flap and #974 roman numerals change Kokoro
output for the same text. Probes that pin phoneme strings may move. **Verify:** P10 Kokoro rows; compare
`ANE` / `ANE-zh` phonemes to the P1 baseline. **Status:** pending.

### R4 — Parakeet long-audio output changes (LOW)
**Problem:** #1032 changes Japanese transcripts on audio > 15 s (fewer dropped tokens); #975 shifts v3/ja seam
word timings (Meetings segment times). Short clips unchanged. **Verify:** P10 `fa-parakeet-tdt-v3`,
`fa-parakeet-ctc-ja`. **Status:** pending.

### R5 — A12 devices on the legacy runtime (LOW, information)
**Problem:** #1009 documents an iPhone XR where Vocoder loads but synthesis fails; no minimum chip is
established. The app's iOS 18.6 floor includes A12. The app's `.cpuOnly` retry only covers load failures.
**Verify:** A12 device. **Status:** needs device; no change in this merge.

### R6 — Cohere `.all` memory blow-up (LOW, dormant)
**Problem:** #1046 shows `.all` routes the INT8 encoder to GPU (12.4 GB footprint on M5 Pro).
`FluidAudioASR` passes `.all` when not `forceCPU`. No live tier has `fa-cohere-transcribe`. **Fix if it ships:**
pass `computeUnits: nil` (new ANE default) or `.cpuAndNeuralEngine`. **Status:** dormant.

### R7 — Trait drift (LOW)
**Problem:** a future upstream default-trait change could drop TTS/Diarizer silently. **Verify:** the app build
compiles `FluidAudioKokoroAneSpeaker` and `FluidAudioDiarizer` (any `#if TTS`-gated symbol missing fails the
build). **Status:** P5 build running.

No other `ModelNames` constant used by a live row changed (P3); no catalogue `components` change is needed
beyond R2.

---

## 4. Fork patches

| File | Resolution |
|---|---|
| `Package.swift` | Upstream's `LuxTtsG2pResources` split; Swift 5 pin on the FluidAudio target beside `.define("TTS")` / `.define("Diarizer")` |
| `Package@swift-6.2.swift` | Kept TTS/Diarizer traits, default `["TTS", "Diarizer"]`; `NemoTextProcessing` stays off (libgit2 modulemap collision, tag-20260918 §3.2) |
| `ModelNames.swift` | Both: `Repo.overrideFolderNames` + upstream `sweepsRootMetadata` (#1047) |
| `TTS/G2P/MultilingualG2PModel.swift` | On upstream `kokoroDirectory()`: nested `<root>/kokoro/`, then flat explicit dir, then flat override root |
| `TTS/KokoroAne/KokoroAneManager.swift` | Upstream `prepareEnglishFrontend()`; fork "load G2P first, skip `ensureG2PAssets` if it loads" gate ported onto `englishG2P` |

**Inventory:** 13 files, **40 → 42** markers by the P3 audit (raw `git grep -c "wangqi modified"` over
`Sources/` + manifests: 42 at `b5cc7a40` → 44 at `97c26284` → **49** at `59d2b4ae`). Nothing disappeared;
G2P, KokoroAneManager and manifest markers were re-dated 2026-10-24 where re-ported. `PocketTTS skipDownload`,
the Kokoro explicit-directory short-circuit, `VadManager` external directory, the EOU per-utterance fix and the
`KokoroAneVariant` per-variant switch are intact.

### Re-ported / new in this upgrade
- **#1025 gates** (`KokoroAneResourceDownloader.swift`, 7 → 9 markers in the merge): `ensureLexiconCache`
  (es/fr) and `ensureMultilingualG2PAssets` gate on `TtsCacheDirectory.overrideDirectory` alone, check nested
  `<directory>/kokoro/` then flat, then **throw**, never download.
- **`59d2b4ae` Mandarin offline gate** (+5 markers, 9 → 14 in the file): with the override set,
  `ensureMandarinG2P` reads `<ANE-zh>/g2p/` or `<ANE-zh>/assets/` (pinyin) else throws;
  `ensureMandarinJiebaHmm` same lookup, missing → HMM off; `ensureMandarinG2pw` reads `<ANE-zh>/g2pw/`
  (`vocab.txt`, `POLYPHONIC_CHARS.txt`, `g2pw.mlmodelc`), missing → dictionary-only. No download.
- **Mirror** `flyingfishinwater/fluidaudios`: uploaded `kokoro-82m-coreml/ANE-zh/g2pw/vocab.txt` and
  `ANE-zh/assets/jieba_hmm_{emit,start,trans}.bin` (sizes match upstream); same four files staged in
  `/Volumes/ssd2t/fluidaudio/FluidInference_kokoro-82m-coreml/ANE-zh/`.

---

### Dependencies

No dependency move. NemoTextProcessing stays disabled.

---

## 5. `libs/audio/fluidaudio/` audit

What the app calls, checked against the merged sources. **No forced consumer fix**: every call compiles unchanged.
No app code switches exhaustively over `AsrModelVersion` or `KokoroAneVariant`. `FluidAudioModelManager` does not import FluidAudio.

| Symbol the app calls | Change | Our usage / fix |
|---|---|---|
| `AsrModels.loadLocal(from:version:)`, `AsrManager.transcribe(_:decoderState:)` | Unchanged signature; `AsrModelVersion` gained `.phonon2` | No exhaustive switch in the app; none |
| `CoherePipeline.loadModels(encoderDir:decoderDir:vocabDir:decoderVariant:computeUnits:)` | Added `encoderComputeUnits` / `decoderComputeUnits` (defaulted); `computeUnits` became optional; default decoder `.v3` (already in tag-20260928) | Compiles; explicit `computeUnits` still wins for both models |
| `ParaformerModels.load`, `SenseVoiceModels.load`, `transcribeDetailed`, `StreamingEouAsrManager`, `VadManager(config:modelDirectory:)`, `processStreamingChunk` | Unchanged | none |
| `KokoroAneManager(variant:defaultVoice:directory:computeUnits:)`, `initialize(preloadVoices:)`, `synthesize(text:voice:speed:)` | Init gained `modelStore:` and `version:` (defaulted `.legacy`) | none |
| `KokoroAneConstants.spanishVoices` / `frenchVoices`, `TtsCacheDirectory.overrideDirectory` | Unchanged | none |
| `PocketTtsManager(language:directory:skipDownload:)`, `synthesize(text:voice:temperature:)` / `(voiceData:)`, `PocketTtsResourceDownloader.ensureMimiEncoder` | Unchanged | none |
| `Nemotron3Models.load(config:directory:computeUnits:)`, `ModelNames.Nemotron3` | Unchanged (trait guard only) | none |
| `TextNormalizer.shared` (ITNHelper) | Unchanged; pass-through without NeMo, as before | none |
| `DiarizerError` | Moved to its own file, still public | The app's own `DiarizerError` (`libs/audio/DiarizerProtocol.swift`) shadows it, as before |

---

## 6. New models that would need new app code

| Model | Size | New class? | Upstream API it would wrap |
|---|---|---|---|
| **Phonon-2** ASR (English, iOS 18+) | 321 MB default encoder (+ decoder/joint), LibriSpeech 2.47 % / 4.62 % WER, faster than v3 | **No**, a `FluidAudioASR` branch + `repoMap` + row | `AsrModels.loadLocal(from:version: .phonon2)` |
| **Parakeet Ultra / Redux** (tag-20260928) | ~620 MB / 183 MB encoder | No, same branch | `loadLocal(version: .ultra / .redux)` |
| **Canary-1B-v2 translation** | int4 bundle, peak ~1.9 GB RAM; size above the 700 MB ASR cap likely, unverified | **Yes**, a new ASR class (`FluidAudioCanaryASR`) or a translate mode | `CanaryManager.transcribe(audio:vad:)` with source ≠ target prompt |
| **PocketTTS small 18m/7m/2m** (English, iOS 18+) | packs well under 200 MB; 18m 109 MB RSS | **Yes**, a new speaker (`FluidAudioPocketTTSSmallSpeaker`) or a second manager type inside `FluidAudioPocketTTSSpeaker` | `PocketTtsSmallManager.load(from:)` (local, no download) + `synthesize` / `synthesizeStreaming` |
| **Gradium Phonon 31m** (English, Apache-2.0) | 171 MB runtime subset | Same as above | `PocketTtsSmallManager` with `.phonon31m` |
| **Paradee-8M** [beta] | 12 MB int8 / 34 MB fp32, English `af_heart` | **Yes**, new speaker (`FluidAudioParadeeSpeaker`) | `ParadeeManager.synthesize` |
| **Kokoro ANE-v3** (en/ja/zh) | not stated; pinned upstream downloads | No new class, a `version: .v3` path in `FluidAudioKokoroAneSpeaker` + new fork gates for `KokoroAneV3Assets` | `KokoroAneManager(version: .v3)` |
| Supertonic-3 v2 | — | Yes (no Supertonic wrapper) | `Supertonic3*` |

Opportunities (not built): diarizer enrollment embeddings (#1018, #1015) for named speakers in Meetings;
Phonon-2 as a faster English Parakeet; Kokoro v3 Japanese; streaming small PocketTTS for low-latency read-aloud.

---

## 7. Deliberately not adopted

| Upstream addition | Why not |
|---|---|
| Kokoro ANE-v3 runtime | Opt-in, new download paths without fork gates, unvalidated on physical iOS devices |
| Canary translation, Phonon-2, small PocketTTS, Phonon 31m, Paradee, Supertonic-3 v2 | Need new app code / catalogue rows (§6); out of scope for an engine upgrade |
| CTC keyword fixes, Nemotron multilingual streaming fixes | Not wrapped |
| Diarizer embedding APIs | No consumer yet |
| tvOS / visionOS builds | Not app platforms |
| Disabling TTS/Diarizer traits | The app uses both |

---

## 8. Verification

| Check | Result |
|---|---|
| `swift build` (macOS), fork worktree after merge (P3) | `Build complete!`, exit 0 |
| `swift build` after `59d2b4ae` (P3b) | `Build complete!`, exit 0 |
| P1 baseline (`run_fluidaudio_tests.py`) | 17 pass; standing FAIL `fa-silero-vad` (doc OPEN) and `fa-kokoro-82m:ANE-zh` (to be fixed by `59d2b4ae` + mirror uploads) |
| App builds (iOS sim + macOS) | macOS BUILD SUCCEEDED (P5, slot 9); iOS build succeeded (P10, slot 9, 487 s) |
| `FluidAudioCatalogueSchemaTests`, `FluidAudioASRDispatchTests` | 11/11 passed (P10) |
| `run_fluidaudio_tests.py --compare <P1 report>` | no REGRESSED. Sweep 14/5/0 vs baseline (fixed: `fa-parakeet-tdt-v3`); the 4 Kokoro rows then passed 4/0/0 after the stage was moved to `components.revision` 4, **including ANE-zh** (standing pre-merge FAIL, fixed). Effective 18/1/0; only `fa-silero-vad` standing (doc OPEN) |

**macOS cannot verify:** the iOS 27 Kokoro advisory (R1: `isBnnsCrashProneOS` is false on macOS 27); #1038 was
verified only on macOS 27 and only for v3; A12 behaviour (R5); background (ANE-only) behaviour of every row.
