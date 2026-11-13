# FluidAudio Upgrade Notes: tag-20261030 → tag-20261113

**Merged:** 2026-11-13 · **Upstream:** FluidInference/FluidAudio `main` (**61 commits**, up to `d653a34a`,
2026-11-11), podspec **v0.18.0 → v0.18.3** · **Merge:** `c4f36c1b` (old fork head `dcf2ea9b`)
**Integration doc:** `helper/docs/audio-fluidaudio.md` · **Consumers:** `libs/audio/fluidaudio/`,
`libs/audio/voiceclone/fluidaudio/VoiceCloneFluidPocketBuilder.swift`, `libs/audio/ITNHelper.swift`,
`SpeechHelper.createFluidAudioSpeaker` / `makeJobSpeaker`, `views/audio/fluidaudio/FluidAudioModelManager.swift`

Most of the 61 commits are quality fixes on paths the app already runs. Long-form Parakeet v3 / Japanese
decoding gets several seam and opening-speech repairs. Cohere cuts Japanese and Chinese long audio at pauses.
PocketTTS sentence chunking is smarter. Kokoro ANE Mandarin follows misaki v1.1 more closely. **One
ship-blocker was found in the tests, not in shipped behaviour (#1089):** `PocketTtsLanguage.english` now names
Kyutai's new `english_2026-10` checkpoint (`v2.1/english_2026-10`). The April pack the
`fa-pocket-tts-english` row ships, in `v2.1/english`, is now `.english2026Apr`. The app now maps the row to
`.english2026Apr` (option A), so users keep the files they already have; adopting 2026-10 is a P8 candidate.
No fork patch was lost (P3: 13 source files / 55 markers, unchanged). No new network path is reachable from a
shipped row. `Package.swift` is unchanged.

---

## 1. Upstream changes, by area

### 1.1 ASR, the shipped paths (`AsrManager` batch v3 / tdtJa, `StreamingEouAsrManager`, Cohere, Paraformer, SenseVoice)

| Commit | What it does | Reaches the app? |
|---|---|---|
| `8ff96513` #1056 | Long-form v3: the post-merge repair also probes the span before the first token, so lost opening speech is recovered | **Yes**: `fa-parakeet-tdt-v3` with audio longer than one window |
| `b20d2098` #1061, `f59050ed` #1097, `30379135` #1128, `e000dfde` #1111 | Seam-gap repair probes every inter-word gap ≥ 0.5 s (was 1.5 s). A window that opens mid-word joins the previous text. A re-spelled word at a seam replaces the kept one. Pending tokens are flushed at an interior window ending in a pause | **Yes**: v3 long-form text at window seams. More probes, so long files decode a little slower |
| `b3eda843` #1098, `70f44050` #1125, `eff40bde` #1143, `a99f5e4a` | tdtJa long-form: no mel context by default; only gaps ≥ 1.5 s are probed; the seam re-decodes a time tail and drops duplicate kept tokens | **Yes**: `fa-parakeet-ctc-ja` long audio |
| `9c9f630b` #1134, `b7610fcf` #1021 | `ASRConfig.resamplerQuality` (default `.balanced`); `AudioConverter(resamplerQuality:)`; streaming file conversion without the 4 GiB cap | No: the app resamples to 16 kHz itself and calls `transcribe(samples)`; all new parameters are defaulted |
| `a31f1515` #1136 | EOU `rearm(keepTranscript:)` (opt-in). Buffered-chunk drains now run one at a time; `reset()` waits for a running drain | **Partly**: `fa-parakeet-eou-120m`. `rearm` is not called. The drain serialization runs on every `process` / `finish` / `reset` and is bounded (a cancelled drain throws at the next chunk) |
| `9a404d63` #1109 | Raw encoder-outputs callback on streaming managers | No (opt-in) |
| `ba90110a` #1146 | Cohere `transcribeLong`: Japanese / Chinese audio is cut into 10–20 s chunks at pauses, joined without a separator | **Yes**: the Cohere row with `ja` / `zh` audio over 35 s |
| `b5813507` #1082, `d11c64d3` #1096 | SenseVoice emotion / event / language confidence from the query logits; pads to the lengths the loaded encoder accepts (fp32) | Compiles unchanged. The app reads `transcribeDetailed` text + language only; it ships int8 |
| `e1bc22be` #1120 | Currency symbols are script-neutral in `TokenLanguageFilter` | v3 with a language hint only |
| vocab: `66a239a3`, `5fff83ae`, `ca9fece9`, `bc025775`, `02bcb9a4`, `9e07b657`, `9e6723c5`, `8303caeb`, `608ec6a7` | Vocabulary boosting / CTC rescoring fixes | No (no custom vocabulary) |
| `653386fa` #1142, `a36629c2` #1145 | Nemotron multilingual: opt-in split 2240 ms encoder, `encoder_v2.mlmodelc` for M2/M3 ANEs | No (§6) |

### 1.2 TTS

| Commit | What it does | Reaches the app? |
|---|---|---|
| `a3303b54` #1089 | **`.english` = `v2.1/english_2026-10`** (new weights, voices incl. `daan`, temperature 0.3, re-projects cloned voices). The April pack becomes `.english2026Apr` (`v2.1/english`). All synthesize / session APIs take `temperature: Float? = nil` → `language.defaultTemperature` | **Yes, handled**: the app maps `fa-pocket-tts-english` → `.english2026Apr` (§5). It passes its own temperature (0.7), so the default change does not reach it |
| `16d0a174` #935, `da7a72ce` #1122, `9599521c` #1121, `22f0ffb1` #1132 | PocketTTS chunking: oversized sentences cut at more punctuation with balanced word splits; no sentence end at a period inside a word or at a decimal; closing quotes stay with their sentence | **Yes**: all 6 `fa-pocket-tts-*` rows; better pauses and fewer mid-number cuts |
| `e43ff1a5` #1058 | Session-lifetime test + docs | No |
| `810a0bd7` #1137, `e63b5230` #1138 | `KokoroAneVocoderPrecision`: opt-in fp32 `KokoroVocoder_v2.mlmodelc` fixes a robotic first ~1 s of long utterances on SME chips (A19 Pro, A20, M5) on the Core ML CPU path. Default stays fp16 | **No change by default** (the app passes nothing). The bug it fixes **does affect the app** on those chips: an opportunity (§7) |
| `d98ec82b` #1131 | `KokoroAneManager.setPunctuationPauses(_:)` | No: not called, so `KokoroAnePauses.spans` returns `[]` and the chain is unchanged |
| `d2bfd78d` #1129, `4b53afa7` #1130, `74f2ed5a` #1055 | Kokoro Mandarin G2P: misaki v1.1 spaces, word slashes, erhua R; g2pW keeps to the mainland dict; word order around a custom-lexicon hit. `/` is now a chunk boundary | **Yes**: Kokoro `ANE-zh` voices; Mandarin phonemes differ |
| `ee78635d` #1104 | Kokoro ANE-v3 follows the OS 27 default routing | No (the app runs the legacy chain) |

### 1.3 Diarization, speaker, download, other

- `d28bde8d` #1081 offline diarizer keeps overlap; `41c53ce4` #1091 Sortformer predicts every frame;
  `392ff8f6` #1148 drains Core ML outputs per batch. **No**: the app diarizes with `Nemotron3Diarizer` only.
- `00efd686` #1099, `d3aab8fa` #1094 CAM++ `_v2` models + per-utterance fbank mean. **No** (the app has no CAM++).
- `fc74bdff` #1150 `ModelCache.validateCompiledModelLayout` now requires a complete `model.mil` + every
  `@model_path/` weight (or `model.espresso.net`, or pipeline stages); a size mismatch re-downloads.
  `96b25669` #1090 resumes interrupted downloads instead of purging. **No**: both live on `ModelHub` / the
  managed cache; the app's directory loaders (`loadLocal`, `ParaformerModels.load`, `SenseVoiceModels.load`,
  the Kokoro / PocketTTS flat paths) do not call them. P3 re-ran the stricter check over every shipped flat
  folder in `/Volumes/ssd2t/fluidaudio/`: all real bundles pass.
- `4e8b59d1` #1083 LFM3 Core AI metadata; podspec 0.18.1–0.18.3; showcase / README / CI commits. Nothing.

---

## 2. What users get from this merge

- **Long Parakeet transcriptions lose fewer words**: opening speech is recovered, and short pauses at window
  seams are repaired (v3 and Japanese).
- **Long Japanese and Chinese Cohere transcriptions keep whole sentences**: audio is cut at pauses.
- **PocketTTS reads long sentences with better pauses**: no breaks inside numbers, abbreviations or quotes.
- **Kokoro Mandarin reads more naturally**: word boundaries, erhua and mainland readings.
- English PocketTTS voices and voice clones sound exactly as before (the April pack is kept on purpose).

## 3. Risk assessment — 6 risks (1 high, 2 medium, 3 low)

### R1 — iOS 27 Kokoro BNNS advisory still in force; macOS cannot verify it (HIGH, unchanged)
**Problem:** `isBnnsCrashProneOS` changed only in its comment; it is still **false on macOS 27** and true on every
non-macOS OS ≥ 26.4, so a green macOS row says nothing about iOS 27 (#889 libBNNS `vadd_fp16_sme_internal`;
#1103 MPSGraph abort on the GPU route). `KokoroAneComputeUnits` (`aneTailCpu` on 27) and the default fp16
vocoder are unchanged. The chain gained a no-op pause hook and a range-based chunker. **Verify:** physical
iOS 27 SME device, `FluidAudioKokoroAneDeviceSoakTests` (`KOKORO_SOAK=1`, about 70 min). **Status:** the
2026-10-30 soak passed (iPhone 16 Pro Max, iOS 27.2, `aneTailCpu`, 1 348 exports). The route did not change
in this range, so a re-soak is advised, not required.

### R2 — PocketTTS English rename (MEDIUM, fixed in the app)
**Problem:** with the old mapping the app and the schema test asked for `.english` = `v2.1/english_2026-10`.
The app still loaded the April files (flat, `skipDownload: true`), but cloned voices took the new reprojection
branch: a missing-assets warning, and the regression probe ran at 0.3.
`testPocketTTSSubpathsMatchTheEnginesOwnLayout` failed. **Fix:** row → `.english2026Apr`; the test and the
probe resolve the language by engine layout (§5). **Verify:** `FluidAudioCatalogueSchemaTests`; P10
`run_fluidaudio_tests.py` `fa-pocket-tts-english` (plain + cloned pass). **Status:** pending P10.

### R3 — Parakeet long-form decoding changes (MEDIUM)
**Problem:** #1056 / #1061 / #1098 / #1111 / #1125 / #1128 / #1143 change which text survives at seams. The gap
floor drops from 1.5 s to 0.5 s, so there are more probe decodes. **Verify:** P10 `fa-parakeet-tdt-v3` and
`fa-parakeet-ctc-ja` against the P1 transcripts (the probes are short, so seam logic is barely exercised);
a long Meetings recording on device. **Status:** pending P10.

### R4 — EOU drain serialization (LOW)
**Problem:** #1136 makes `reset()` wait for a running drain. `cancelStreaming()` → `reset()` after a feed was
cancelled: the drain throws at its next `checkCancellation`, so the wait is bounded by one chunk. **Verify:**
P10 EOU row; on device, cancel keyboard dictation mid-utterance, then dictate again. **Status:** audited.

### R5 — PocketTTS chunker rewrite (LOW)
**Problem:** #935 / #1121 / #1122 / #1132 change chunk boundaries, so durations shift a little. **Verify:** P10
6 PocketTTS rows (median-of-3 duration band). **Status:** pending P10.

### R6 — Kokoro Mandarin phonemes (LOW)
**Problem:** #1129 / #1130 / #1055 change zh phonemes, and `/` is a chunk boundary. **Verify:** P10
`fa-kokoro-82m:ANE-zh`. **Status:** pending P10.

---

## 4. Fork patches

| File | Resolution |
|---|---|
| `TTS/KokoroAne/Assets/KokoroAneResourceDownloader.swift` | Upstream added `additionalModelNames` (fp32 vocoder); the explicit-directory short-circuit still runs first |
| `TTS/PocketTTS/Assets/PocketTtsResourceDownloader.swift` | Gate is now `reprojectsSharedEncoder`; the flat return and the `skipDownload` throw both still come before any fetch |
| `TTS/PocketTTS/Pipeline/PocketTtsModelStore.swift`, `PocketTtsManager.swift` | `isEnglish`; temperature default `nil` → pack default. Patches intact |
| `ASR/Parakeet/Streaming/EOU/StreamingEouAsrManager.swift` | Both per-utterance EOU markers intact beside upstream's opt-in `rearm()` and serialized drain |

**Inventory:** 13 source files, **55 → 55** markers (P3, `git grep -c "wangqi modified"` at `dcf2ea9b` vs the
merge). The other 8 patched files (`ModelNames` overrides, G2P, MultilingualG2P, VadManager, TtsCacheDirectory,
PocketTtsSmall) are not in the merge diff. Nothing absorbed, nothing added.

### Dependencies

No dependency move. `Package.swift` / `Package@swift-6.2.swift` unchanged (Swift 5 mode, TTS/Diarizer traits on).

---

## 5. `libs/audio/fluidaudio/` audit

| Symbol the app calls | Change | Our usage / fix |
|---|---|---|
| `PocketTtsManager(language:directory:skipDownload:)` | `.english` now = `v2.1/english_2026-10` | **Fixed**: `FluidAudioPocketTTSSpeaker.pocketTtsLanguage(from:)` returns `.english2026Apr` for `fa-pocket-tts-english` (was `.english`) |
| `PocketTtsManager.synthesize(text:voice:/voiceData:temperature:)` | `temperature: Float?` (nil → pack default) | none: the app passes its setting (default 0.7) |
| `PocketTtsResourceDownloader.ensureMimiEncoder`, `cloneVoice`, `save/loadClonedVoice` | `.english2026Apr` uses the shared encoder, no reprojection (same as the old `.english`) | none |
| `KokoroAneManager(variant:defaultVoice:directory:computeUnits:)`, `initialize`, `synthesize(text:voice:speed:)` | New defaulted `vocoderPrecision:` (.float16); no-op pauses; range chunker | none |
| `AsrModels.loadLocal(from:version:)`, `AsrManager()`, `transcribe(_:decoderState:)` | Long-form seam / opening repairs; `ASRConfig` defaulted additions | none |
| `StreamingEouAsrManager` `reset` / `process` / `finish` | Serialized drains; `rearm` opt-in | none |
| `CoherePipeline.transcribeLong` | ja / zh pause chunks | none |
| `ParaformerModels.load`, `SenseVoiceModels.load`, `SenseVoiceManager.transcribeDetailed` | SenseVoice tag scores, padding | none |
| `Nemotron3Models` / `Nemotron3Diarizer`, `VadManager(modelDirectory:)`, `PocketTtsSmallManager`, `TextNormalizer` | Unchanged | none |

No app code switches exhaustively over `PocketTtsLanguage` (new `.english2026Apr`) or the new
`KokoroAneVocoderPrecision` / `AudioConverter.ResamplerQuality`. `FluidAudioModelManager` does not import
FluidAudio; `SpeechHelper.createFluidAudioSpeaker` / `makeJobSpeaker` dispatch by model id and are unaffected.

**Tests changed:** `FluidAudioCatalogueSchemaTests.testPocketTTSSubpathsMatchTheEnginesOwnLayout` resolves the
pack as the one `PocketTtsLanguage` whose `repoSubdirectory` is the subpath's tail. Exactly one must match,
so a wrong or ambiguous subpath still fails. `FluidAudioCatalogueRegressionTests.probePocketTTS` maps
`FLUIDTEST_LANGUAGE` (the pack folder) the same way, and runs the cloned pass for both English cases.

---

## 6. New models that would need new app code

| Model | Size | New class? | Upstream API it would wrap |
|---|---|---|---|
| **Kyutai PocketTTS English 2026-10** (#1089) | one `v2.1/<lang>` pack (same layout as April) + root `encoder_recover_pinv.bin` | **No**: a new row (`fa-pocket-tts-english-2026-10`) + a `-2026-10` suffix in `pocketTtsLanguage(from:)` | `PocketTtsManager(language: .english, …)`; cloning re-projects (`speaker_proj_weight.bin`, `encoder_recover_pinv.bin`) |
| **Nemotron multilingual streaming** (`encoder_v2` #1145, split encoder #1142) | per tier: `encoder_v2.mlmodelc` in `encoder-v2/<tier>ms/` | **Yes**: `FluidAudioNemotronStreamingASR` (the app has no Nemotron streaming wrapper) | `StreamingNemotronMultilingualAsrManager`, `loadModels(from:splitEncoderDirectory:)` |
| **CAM++ `_v2`** (#1099) | `CamPlusPreprocessor_v2` + `CamPlusPlus_v2` | **Yes**: a speaker-embedding class (e.g. `FluidAudioSpeakerEmbedder`) if speaker ID is ever wanted | `CampPlusEmbedder` / `CampPlusModels` |

Carried from tag-20261030, still not built: PocketTTS small 18m/7m/2m (rows only, now that
`FluidAudioPocketTTSSmallSpeaker` ships `fa-pocket-small-phonon-31m`), LFM3-Audio-2B (over cap),
Phonon-2.5 and Bark 3.0 / 3.0 Clean (`FluidAudioASR` branch), Nemotron Unified 0.6B (`FluidAudioUnifiedASR`),
Nemotron 3.5 Lite (over cap), Paradee v1.2 (`FluidAudioParadeeSpeaker`), Canary, Kokoro ANE-v3, Supertonic-3 v2.

## 7. Opportunities (not built)

| Upstream feature | Consumer that would use it | Value |
|---|---|---|
| `vocoderPrecision: .float32` (`KokoroVocoder_v2.mlmodelc`, 47 MB per variant folder) | `FluidAudioKokoroAneSpeaker` on SME chips | Fixes the robotic first second of long utterances on A19 Pro / A20 / M5. The vocoder takes ~2.8x longer (0.53 → 1.49 s per 23.6 s of audio on an iPhone 17 Pro Max). Needs the file in `ANE/`, `ANE-ja/`, `ANE-zh/` on the mirror and a device soak |
| `setPunctuationPauses(_:)` | Kokoro Mandarin (`、` `，` get 50–125 ms) | Natural pauses for zh long-form / Text to Audio |
| EOU `rearm(keepTranscript:)` | `FluidAudioASR` streaming (keyboard dictation) | Could replace the fork's per-utterance EOU patch with the upstream API |
| SenseVoice emotion / event tags | Meetings | Emotion / laughter / applause markers |
| PocketTTS English 2026-10 | new row | Reads sentence-final numbers and years; new voice `daan` |

## 8. Deliberately not adopted

| Upstream addition | Why not |
|---|---|
| `.english` → `english_2026-10` for the existing `fa-pocket-tts-english` row | It would change what an installed row means: the files on disk are the April pack, while `.english` asks for 2026-10 behaviour (temperature 0.3 by default, re-projected clones that need assets the row does not ship). New weights would need new mirror files and a re-download. Kept on `.english2026Apr` (option A). **Option B, a candidate for P8:** an additive row `fa-pocket-tts-english-2026-10`, `repo_subpath pocket-tts-coreml/v2.1/english_2026-10`. Mirror additions (none uploaded yet): `v2.1/english_2026-10/{cond_prefill,flow_decoder_fused,flowlm_step,mimi_decoder}.mlmodelc`, `constants_bin/*` (incl. `speaker_proj_weight.bin`, voice `daan`), `manifest.json`, and the root `pocket-tts-coreml/encoder_recover_pinv.bin`. Needs a quality check vs April |
| fp32 Kokoro vocoder | Opt-in upstream; needs mirror files and an iOS 27 soak (§7) |
| Nemotron multilingual encoder_v2 / split, CAM++ v2, vocabulary boosting fixes, raw encoder callback, resampler quality | No consumer in the app |

---

## 9. Verification

| Check | Result |
|---|---|
| P3 patch audit | 13 files / 55 markers intact; no new network path on a shipped row; `ModelNames` change on a shipped row only via `PocketTtsLanguage.english` (handled) |
| `swift build` (fork, P3 and P6) | Build complete, 0 errors |
| App builds (P5 macOS, P10 iOS) | pending |
| `FluidAudioCatalogueSchemaTests`, `FluidAudioASRDispatchTests` | pending (the schema suite fails without the §5 test change) |
| `run_fluidaudio_tests.py --compare <P1>` | pending P10 |
| iOS 27 Kokoro soak | not re-run; the route is unchanged since the 2026-10-30 PASS (R1) |

**macOS cannot verify:** the iOS 27 Kokoro BNNS advisory (R1); the SME robotic-onset bug (A19 Pro / A20 / M5,
also M5 Macs); keyboard dictation cancel on device (R4); background (ANE-only) behaviour of every row.
