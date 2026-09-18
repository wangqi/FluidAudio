# FluidAudio Upgrade Notes: tag-20260509 → tag-20260918

**216 commits, 2026-05-09 → 2026-09-18** (55 `feat`, 82 `fix`, 3 `perf`, 4 `refactor`, 4 `chore`).
284 source files changed: +35,657 / −17,499.

This is the largest upgrade the fork has absorbed. It removes four backends the app depended on,
rewrites the whole download stack, and introduces an **iOS 27 crash advisory that affects the
engine the app now relies on for Kokoro TTS**. Read the Risks section before shipping.

---

## 1. Breaking removals

| Removed upstream | Commit | App code that used it | Resolution |
|------------------|--------|----------------------|------------|
| `DownloadUtils` → `Shared/Download/*` (`ModelHub`, `ModelCache`, `HFClient`, `FileDownloader`, `HFTreeLister`, `RetryPolicy`, `ProgressReporter`, `DownloadTypes`) | `refactor(download)!: Wave 6 — ModelHub replaces DownloadUtils (#765)` | `VadManager` fork patch; `FluidAudioASR` comments | Fork patches re-homed onto `ModelHub.loadModels`. `DownloadUtils.ProgressHandler` → top-level `ProgressHandler`; `DownloadUtils.HuggingFaceDownloadError` / `.OfflineError` → one merged `DownloadError` |
| Standard CoreML Kokoro — 13 files (`KokoroTtsManager`, `TtsModels`, `KokoroSynthesizer`, `TtsResourceDownloader`, `KokoroVocabulary`, `KokoroChunker`, `TtsTextPreprocessor`, `TtsCustomLexicon`, …) | `deprecate: remove CosyVoice3 and mono Kokoro (#571)` | `FluidAudioKokoroSpeaker` | Speaker deleted. Its catalog, `tts_fluidKokoro_voice` and `tts_fluidKokoro_speed` moved onto `FluidAudioKokoroAneSpeaker` |
| `Qwen3AsrManager` + 5 files | `chore(asr): remove experimental Qwen3 ASR backend` | `FluidAudioASR` `fa-qwen3-asr` branch | Branch and catalog row removed |
| `CtcZhCnManager` / `CtcZhCnModels` | `chore(asr): remove experimental Parakeet CTC zh-CN Mandarin model` | `FluidAudioASR` `fa-parakeet-ctc-zh-cn` branch | Branch and catalog row removed. Upstream's Mandarin replacements are `ParaformerManager` / `SenseVoiceManager` — **not adopted** |
| Magpie TTS (22 files), CosyVoice3 TTS (20 files) | `Remove experimental Magpie multilingual TTS backend`, `deprecate: … (#571)` | — | Never used by the app |

`TtsModels.overrideCacheDirectory` — the fork's central hook for pointing FluidAudio at the app's
flat download folder — died with `TtsModels`. It now lives on upstream's new
`TtsCacheDirectory.overrideDirectory` (`TTS/Shared/TtsCacheDirectory.swift`).

---

## 2. What the upgrade brings that matters to this app

### 2.1 Kokoro ANE inherited the full voice catalog

`feat(tts/kokoro-ane): any Kokoro-82M v1.0 voice for the English variant (#896)` is what makes the
removal of the standard chain survivable. The ANE variant now serves **54 English voices** (up from
`af_heart` alone), 103 Mandarin, and 5 Japanese. English voices other than `af_heart` are published
as repo-root `voices/<name>.json` and converted to the flat fp32 `.bin` on first use.

Related quality work on the English frontend, all landed in this range:

- `fix(tts): Misaki-lexicon-first English frontend for KokoroAne` — lexicon lookup before BART G2P.
- `fix(tts/kokoro-ane): read uppercase initialisms as letter names (#710)` — "NASA" → letter names.
- `fix(tts/kokoro-ane): smart-apostrophe contractions + hyphenated lexicon lookups (#774, #775)`.
- `fix(tts/kokoro-ane): stem possessives instead of G2P-ing the whole token`.
- `fix(tts/kokoro-ane): treat quote delimiters as punctuation`.
- `feat(tts): shared English text normalization for raw numbers/times (#711)`.
- `feat(tts): auto-chunk long text in KokoroAne high-level synthesize (#712)` — **landed then
  reverted** inside this same range. `feat(tts/kokoro): byte-exact NeMo text normalization before
  G2P (#790)` removed the `PhonemeChunker` call and the `synthesizeChunks` helper from
  `KokoroAneManager`, and restored the header doc line *"IPA input capped at 512 tokens — chunk
  longer prompts upstream."* `synthesize(text:voice:speed:)` still runs one chain and still throws
  `KokoroAneError.phonemeSequenceTooLong` on over-cap input, so **the app's own chunker in
  `FluidAudioKokoroAneSpeaker.splitIntoChunks` remains required.**

### 2.2 Audio-quality fixes to the ANE chain

- `fix(tts/kokoro-ane): KokoroNoise v2 — atan2 phase fix (removes HF sharpness)`.
- `fix(tts/kokoro-ane): adopt COLA-corrected KokoroTail_v2 + native output level for all variants`.
- `fix(tts/kokoro-ane): throw instead of trapping on non-finite PostAlbert durations` — a `fatalError`
  class turned into a catchable throw.
- `Fix KokoroAne strided MLMultiArray handling`.

### 2.3 Download stack rewrite (`#765`, Waves 2–6)

- `feat(download): resume interrupted downloads with HTTP Range requests` — a dropped transfer
  continues from the bytes on disk instead of restarting.
- `feat(download): stall watchdog + DownloadConfig plumbing (#810)` — `minStallBytes` (default 1 MiB)
  / `stallWindow` (default 120 s) surface a frozen CDN connection in seconds rather than at the
  30-minute idle timeout.
- `fix: preserve model cache when first load is cancelled` and
  `fix(download): preserve model cache on transient network errors` — cancellation and transient
  failures no longer delete a fully-downloaded multi-hundred-MB repo.
- `perf(download): skip unused PocketTTS variants + concurrent subdirectory fetches` —
  `maxConcurrentFiles` (default 4).
- `Validate downloaded model artifacts before caching (#740)`,
  `feat(download): byte-weighted progress for downloadSubdirectory`,
  `fix(download): deliver byte-level progress during download (#756)`.
- `feat(download): add offline-only enforcement` → today's `ModelHub.offlineMode`.

### 2.4 iOS storage location

`TTS: store downloaded models in Application Support, not Caches (iOS) (#642)` — every TTS
downloader used `Library/Caches/`, which the system reclaims under disk pressure, silently purging
hundreds of MB while the app is backgrounded. Now `.applicationSupportDirectory`, matching the ASR
side. **No impact on this app**, which stages every model itself under `models/fluidaudio/` via
`DownloadManagerCoreML` and never lets FluidAudio choose a path — but it removes a latent trap for
the default path used by the shared G2P assets.

### 2.5 PocketTTS

- `PocketTTS v2.1: fused flow decoder (ANE) + cond prefill + fp16 flowlm (~1.8× RTFx)`.
- `feat(tts/pocket): ANE placements — rank-4 split-KV models (.ane) + MLState pipeline (.aneState)`.
- `feat(tts/pocket): per-stage compute-unit overrides (#881)` — `PocketTtsComputeUnits`, including
  `.avoidNeuralEngine` for hardware where a stage aborts.
- `feat(tts/pockettts): add 5 native-language voices + slim language-pack downloads ~40%`.
- `fix(tts/pocket-tts): per-language mimi encoders for non-English voice cloning (#793)` and
  `fix(tts/pocket-tts): enable voice cloning for 24-layer non-English packs (#793)` — cloning on the
  French/German/Italian/Portuguese/Spanish packs can now use the pack's own encoder instead of
  reprojecting through the English one. **Not reachable as configured**:
  `ensurePackMimiEncoder` looks for `v2.1/<lang>/mimi_encoderv3.mlmodelc`, which is not in the
  app's `components.required`, and `DownloadManagerCoreML` fetches only listed components. The
  fork's offline gate returns `nil` for it, so cloning falls back to the shared encoder +
  reprojection exactly as before. Adding `mimi_encoderv3.mlmodelc` to the six pocket rows would
  claim it, at the cost of a larger download for every user whether they clone voices or not —
  **left as a decision, not applied.**
- `fix(tts/pockettts): normalize French text and preserve mid-sentence chunks (#584)`.

### 2.6 ASR

Applicable to the models this app ships:

- `perf(asr): opt-in GPU encoder placement for Parakeet v3 (+~8% RTFx, WER-neutral)`.
- `feat(asr/v3): opt-in int8-linear Encoder_v2 encoder precision (#760)`.
- `feat(asr/eou): opt-in fused decoder+joint_decision path (+7-9% RTFx, WER neutral-or-better)`.
- `Add timestamp support to EoU Streaming` / `Timestamping RTTN decoder`.
- `fix(asr/eou): debounce on wall-clock silence, not consecutive EOU emissions` — upstream
  reimplemented the EOU debounce as the pure `evaluateEouDebounce`. **The fork's per-utterance EOU
  patch is still required on top** (see §4, item 10).
- A long run of seam/merge correctness fixes for long-form transcription (`#683`, `#706`, `#825`,
  `#855`, `#897`, `#909`) — chunk-boundary word loss, final-window truncation, blank-decode rescue.
- `fix(asr): fetch parakeet_vocab.json in AsrModels.download (#748)`,
  `fix(asr): resolve sentence-final punctuation ids from the loaded vocabulary (#905)`.

Not applicable (new backends, none adopted): Nemotron 3.5 Multilingual (40 locales),
Parakeet Unified 0.6B, SenseVoiceSmall, Paraformer-large (zh), Canary-1B-v2 [beta].

### 2.7 VAD and diarization

- `Update Silero VAD CoreML artifact to v6.2.1` — the `VadManager` API is unchanged (only
  `DownloadUtils.ProgressHandler` → `ProgressHandler`), but `ModelNames.VAD.sileroVad` was renamed
  from `silero-vad-unified-256ms-v6.0.0` to `…-v6.2.1`. **This silently breaks the app's VAD**:
  `VadManager.loadUnifiedModel` resolves `models[ModelNames.VAD.sileroVadFile]`, so a cache
  staged under the old name yields `nil` and `VadError.modelLoadingFailed`. Fixed by bumping
  `components` for `fa-silero-vad` in all five config tiers; existing installs re-download (~5 MB).
- `feat(vad): FSMN-VAD backend (CoreML) [beta]` — new, not adopted.
- `Fixed LS-EEND Memory Leak + Updated Docs`, `LS-EEND Speaker Pre-Enrollment Bugfixes`.
- `OfflineDiarizerManager: split process() into prepare()/cluster()` — cacheable
  segmentation+embeddings.
- `feat(diarizer): optional progressHandler on performCompleteDiarization`.
- `feat(diarizer): expose per-chunk embeddings on DiarizationResult`.
- `fix(offline-diarizer): pyannote-parity clustering — threshold semantics, constraint count,
  constrained assignment` and `fix(diarization): deterministic & robust offline VBx re-clustering`
  — **offline diarization output will differ from tag-20260509 for the same audio.**
- `fix(diarizer/offline): propagate cancellation to workers`.
- `feat(speaker): CAM++ speaker-embedding backend (CoreML) [beta]` — new, not adopted.

### 2.8 ITN / text normalization

`fix(itn): link the bundled NeMo engine directly instead of dlopen(nil) discovery` changes
`TextNormalizer.isNativeAvailable` from a runtime `dlsym` probe into a **compile-time constant**
driven by the `NemoTextProcessing` package trait. See §3.1 — this is the one change that would
alter app behaviour for free, and the one this fork currently cannot take.

---

## 3. Risk assessment

### 3.1 HIGH — iOS 27: no safe Core ML route for Kokoro ANE

`KokoroAneManager.osAdvisory(for:)` now returns, for **any non-macOS OS ≥ 27**:

> iOS/iPadOS 27: no Core ML route for Kokoro ANE is known to be safe. The default Metal-free route
> (noise + tail on CPU) has crashed in libBNNS (`vadd_fp16_sme_internal` SIGSEGV) after ~1 h of
> synthesis, and the Metal route aborts in MPSGraph within minutes. Both are uncatchable
> in-process. Consider disabling Kokoro ANE on this OS line until a safe route is shown.
> — FluidInference/FluidAudio#889

`isBnnsCrashProneOS` flags **everything from iOS 26.4 onward** (macOS is clear from 26.6, and
macOS 27 is unflagged).

Why this matters more than it did last upgrade: before this merge the app could fall back to the
standard CoreML Kokoro chain, which does not use this route. Upstream deleted it, so
`FluidAudioKokoroAneSpeaker` is now the **only** FluidAudio Kokoro engine. On an iOS 27 device a
long Text-to-Audio job is a crash candidate, and the crash is a SIGSEGV inside Apple's framework —
`do/catch` cannot contain it.

Mitigations, in order of preference:

1. Ship no FluidAudio Kokoro on iOS 27 — the app already has three other Kokoro paths
   (SherpaONNX, Core AI, MLX Audio). `SpeakerType.visibleCases` is the established mechanism for
   hiding an engine the OS cannot run.
2. Cap exposure: keep it for short read-aloud, exclude it from Text-to-Audio long-form jobs.
3. Ship as-is and accept the crash risk.

This is a product call, not a merge call — **it is not resolved in this branch.**

### 3.2 HIGH — `NemoTextProcessing` cannot be linked (build-breaking)

`feat(package): make NemoTextProcessing an opt-out trait (#880, #888)` adds a prebuilt Rust
staticlib xcframework. It is a **static-library** xcframework, so Xcode unpacks its headers into
`$BUILT_PRODUCTS_DIR/include/` — where the app's `libgit2.xcframework` (SwiftGit2) already puts
its own. Both ship a `module.modulemap`, and the app build fails outright:

```
error: Multiple commands produce '.../Build/Products/Debug-iphonesimulator/include/module.modulemap'
```

The fork sets `.default(enabledTraits: [])` in `Package@swift-6.2.swift` to unblock the build.

**Correction to the first read of this**: because §2.8 turned `isNativeAvailable` into a
compile-time constant, enabling the trait is no longer a no-op. Pre-upgrade the app's `ITNHelper`
always fell through to `SwiftITN` (nothing ever linked a NeMo staticlib, so the `dlsym` probe
always failed). Disabling the trait preserves exactly that behaviour — nothing regresses — but it
also means the app **forgoes a real upgrade**: byte-exact NeMo inverse text normalization in 7
languages for ASR output and the TTS frontends, replacing a hand-rolled Swift fallback that
handles only cardinals, currency, percentages and ordinals.

To claim it: repackage libgit2 as a framework-style xcframework
(`thirdparty/SwiftGit2/build-xcframework.sh`) so the two stop sharing `include/`. Cost is roughly
+8 MB per architecture slice.

### 3.3 MEDIUM — offline diarization results change

`fix(offline-diarizer): pyannote-parity clustering` changes threshold semantics and constrained
assignment; `fix(diarization): deterministic & robust offline VBx re-clustering (K-Means n_init)`
changes initialization. Speaker labels and boundaries for the same recording will not match
tag-20260509. Anything that stored diarization output and compares it across app versions will see
a discontinuity. Upstream frames both as accuracy improvements.

### 3.4 WAS-BROKEN, NOW FIXED — Silero VAD model filename

`Update Silero VAD CoreML artifact to v6.2.1` renamed `ModelNames.VAD.sileroVad` from
`silero-vad-unified-256ms-v6.0.0` to `…-v6.2.1`. `VadManager.loadUnifiedModel` resolves the loaded
bundle by that exact key, so a cache staged under the old name returns `nil` and throws
`VadError.modelLoadingFailed` — FluidAudio VAD would have stopped working entirely after this
upgrade, with no compile error to warn anyone.

Fixed by bumping the `fa-silero-vad` `components` in the four live config tiers
(`helper/fluidaudio_model{,_test,_audit}.json`, `data/fluidaudio_model.json`;
`_global.json` is an orphan with zero references repo-wide and was deliberately left alone).
Existing installs will see the required component missing and re-download (~5 MB).

**Now gated by `FluidAudioCatalogueSchemaTests`**, which asserts the row's components are a
superset of `ModelNames.VAD.requiredModels` — composed from the constant, not copied as a
string, so the next rename is a compile error or a named assertion failure rather than a silent
break. Verified against the live HF tree: `silero-vad-unified-256ms-v6.2.1.mlmodelc` exists in
`FluidInference/silero-vad-coreml` (1.1 MB).

### 3.5 WAS-BROKEN, NOW FIXED — PocketTTS pack path and two model filenames

`PocketTTS v2.1: fused flow decoder (ANE) + cond prefill + fp16 flowlm (~1.8× RTFx)` re-converted
every language pack and changed three things the app's download config pins:

| | tag-20260509 | tag-20260918 |
|---|---|---|
| `PocketTtsLanguage.repoSubdirectory` | `v2/<lang>` | `v2.1/<lang>` |
| conditioner | `cond_step.mlmodelc` | `cond_prefill.mlmodelc` |
| flow decoder | `flow_decoder.mlmodelc` | `flow_decoder_fused.mlmodelc` |

`ModelNames.PocketTTS.requiredModels(precision:placement:)` checks the new names, so a pack staged
under the old ones fails `allPresent`, fails the flat-layout fallback, and — because the app passes
`skipDownload: true` — throws `PocketTTSError.modelNotFound`. **PocketTTS would have produced no
audio at all, in any of the six languages, with no compile error to warn anyone.**

Fixed by updating `repo_subpath` and `components` for all six `fa-pocket-tts-*` rows in the four
live config tiers. Existing installs re-download the pack (~750 MB each). The `size` fields were
left alone; `feat(tts/pockettts): … slim language-pack downloads ~40%` suggests the real figure is
now lower, but upstream publishes no per-pack number — **re-measure on device and correct the
config.**

**Now gated by `FluidAudioCatalogueSchemaTests`** on both axes: the components against
`ModelNames.PocketTTS.requiredModels(precision: .fp16, placement: .gpu)`, and `repo_subpath`
against `PocketTtsLanguage.repoSubdirectory` — the engine's own copy of the path, so the two can
no longer drift apart unnoticed. Verified against the live HF tree: `v2.1/english/` contains
`cond_prefill.mlmodelc` and `flow_decoder_fused.mlmodelc`.

### 3.6 MEDIUM — 54 English Kokoro voices are newly reachable, and only the first is proven

`tts_fluidKokoro_voice` now actually reaches synthesis on the ANE chain, where it previously did
nothing. 53 of those voices have never been exercised through this app. Each non-`af_heart` voice
takes a conversion path on first use (repo-root `voices/<name>.json` → flat fp32 `.bin`) that the
fork patched to work offline; a voice whose JSON was not staged fails at synthesis time, not at
load time. Worth a pass over the voice list on device before release.

### 3.7 RESOLVED — Mandarin ASR replacement is wired

`fa-parakeet-ctc-zh-cn` and `fa-qwen3-asr` are gone from the catalog. **Both upstream
replacements are now adopted**, so this is no longer a feature removal:

| New row | Manager | Size (int8) | Notes |
|---|---|---|---|
| `fa-paraformer-zh` | `ParaformerManager` | 222 MB | Mandarin, **has timestamps** (`transcribeWithTimestamps` → `[TimestampedSegment]`, maps 1:1 onto `ASRSegment`) |
| `fa-sensevoice` | `SenseVoiceManager` | 240 MB | zh/en/yue/ja/ko, auto-detect, text only |

Both ship int8 rather than fp16: 222 MB vs 436 MB and 240 MB vs 473 MB, at an upstream-measured
identical CER (2.12% for Paraformer on AISHELL). Both beat what they replace on size *and*
accuracy — `fa-parakeet-ctc-zh-cn` was 571 MB at 8.2% CER, `fa-qwen3-asr` 600 MB.

Each gets its **own `else if` branch** in `FluidAudioASR.loadManagerIfNeeded`, never the generic
fallthrough: that branch sets `Repo.overrideFolderNames` only for `parakeetV3`/`parakeetV2`, so a
new id falling through it would resolve against the bare HuggingFace slug — the bug recorded in
`helper/docs/plans/fix-fluidaudio-model-files-looked-up-at-wrong.md`. Neither new loader needs
the override at all; both open the `.mlmodelc` bundles directly inside the directory.

**Stored ids self-heal.** `FluidAudioASR.retiredModelMigrations` maps `fa-parakeet-ctc-zh-cn` →
`fa-paraformer-zh` and `fa-qwen3-asr` → `fa-sensevoice` — a directed migration rather than a
reset to the English default, so a Mandarin user lands on the Mandarin replacement.
`healStoredModelId()` **persists** the correction; the pre-existing half-fix in `ASRSettingView`
fell back in the picker but never wrote it back, so `FluidAudioASR.init` kept reading the dead id
from iCloud KV and loading nothing. Covered by `FluidAudioASRDispatchTests`.

### 3.8 LOW — download-stack surface is entirely new code

Every byte the app fetches from HuggingFace for FluidAudio now goes through code written in this
range (resume, stall watchdog, concurrency, artifact validation). It is better-tested upstream than
what it replaces and has explicit regression tests (`DownloadArtifactValidationTests`,
`DownloadCancellationTests`), but it is new. The app's own `DownloadManagerCoreML` does the actual
staging, so the blast radius is limited to the paths the fork deliberately gates with
`skipDownload` / the cache override.

### 3.9 PRE-EXISTING, NOW FIXED — `fa-parakeet-ctc-ja` named a dead repo and the wrong file set

`FluidAudioASR` loads Japanese with `AsrModels.load(from:version:.tdtJa)`, whose
`ModelNames.TDTJa.requiredModels` is `Preprocessor.mlmodelc`, `Encoder.mlmodelc`,
`Decoderv2.mlmodelc`, `Jointerv2.mlmodelc`. The app's config instead required
`Preprocessor.mlmodelc`, `Encoder.mlmodelc`, `CtcDecoder.mlmodelc`, `vocab.json`. `TDTJa` did
**not** change in this range, so this predates the upgrade.

Building the schema gate surfaced the rest of it: the row's `repo` was
`FluidInference/parakeet-ctc-0.6b-ja-coreml`, which **no longer exists** (the HF tree API returns
404), while `Repo.parakeetJa` is `FluidInference/parakeet-0.6b-ja-coreml`. So Japanese ASR could
not work on any fresh install — the download either failed outright or staged CTC files the TDT
loader never opens.

Fixed: the row now points at `FluidInference/parakeet-0.6b-ja-coreml` with the TDT file set
(619 MB measured from the live tree), `cache_folder` and `FluidAudioASR.repoMap` follow, and
`FluidAudioCatalogueSchemaTests` asserts the components against `ModelNames.TDTJa.requiredModels`.
Devices holding the old folder re-download. The transcript itself still has **no oracle** in the
repo — the regression harness records the Japanese row as `oracle: false` rather than implying a
verified result.

### 3.11 PRE-EXISTING, NOW FIXED — `fa-parakeet-tdt-v3` never downloaded its joint decoder

Found by the same schema gate. `AsrModels.load(from:)` defaults to `version: .v3`, and
`getModelFileNames` returns `Names.jointV3File` — `JointDecisionv3.mlmodelc` — for v3
**exclusively**; the unsuffixed `JointDecision.mlmodelc` is the v2/110m/ja file. The catalogue
listed only the unsuffixed one, so a fresh download produced a directory that throws
`AsrModelsError.loadingFailed("Failed to load joint model JointDecisionv3.mlmodelc")` at load.

It went unnoticed because developer machines were staged before `jointV3File` became the v3
default and still have both bundles on disk. Fixed by adding `JointDecisionv3.mlmodelc` to the
row's components (and correcting `size` to 620 MB); gated by `FluidAudioCatalogueSchemaTests`
against `ModelNames.ASR.requiredModelsV3(precision: .int8)`.

**This is the argument for the static tier.** Three of the five breakages in this document are
one catalogue string disagreeing with one engine constant, and none of them produced a compile
error, a failing test, or a runtime error anywhere a developer would see it.

### 3.12 OPEN, PRE-EXISTING — `fa-silero-vad` never loads the bundle we download

Found by the first FluidAudio regression baseline, 2026-09-18. **Not fixed here.**

`VadManager(config:modelDirectory:)` hands the directory straight to
`ModelHub.loadModels(.vad, directory:)`, which resolves
`directory.appendingPathComponent(repo.folderName)`. `FluidAudioVAD` passes the pack root, so
the loader looks in `…/FluidInference_silero-vad-coreml/silero-vad-coreml/` — a directory
`DownloadManagerCoreML` never creates. No `Repo.overrideFolderNames[.vad]` is set, and
`ModelHub` has no flat-root fallback on this path.

Consequence: the app re-downloads the ~1 MB bundle into a nested subdirectory on first use and
then works, so this has been invisible; the selective download we pay for is discarded. On a
read-only model stage it surfaces as an EPERM instead.

Two-line fix, mirroring the Parakeet paths: set `Repo.overrideFolderNames[.vad] = cacheFolder`
and pass the PARENT directory to `VadManager`. Worth confirming against §3.4 before shipping —
the v6.2.1 component bump is correct either way, but nobody has yet seen the app load the
staged file.

### 3.13 MEASURED — Paraformer cannot transcribe past 30 s in one call

The regression ladder measured a ceiling the plan had wrong. `decoderEncFrames = 512` and
`decoderMaxTokens = 128` are real, but they are not what bites first: the PREPROCESSOR rejects
anything outside `3200..480000` samples, so a 39.19 s clip throws
`Size (626960) of dimension (1) is not in allowed range` before the decoder is reached.

`FluidAudioASR` does no chunking for Paraformer, so Mandarin transcription of anything longer
than 30 s currently fails outright rather than truncating. The harness pins this as the 5x
rung's PASS condition, so it will notice if the cap ever moves — but the app-side chunking is
an open product question, not something the gate can fix.

Also measured, for the record: Paraformer returns
`…在北京见证…较长的监督史` for `…在北京建政…较长的建都史` on the test clip — two homophone
substitutions, 13.3% CER on a 30-hanzi sentence. Upstream's 2.12% is a corpus average; a single
short utterance is a much noisier estimate, and the gate's ceiling is set at 20% accordingly.

### 3.10 LOW — Swift 6.2+ manifest shadowing

`Package@swift-6.2.swift` is new upstream and **shadows `Package.swift` on Swift 6.2+ toolchains**.
The fork's `.swiftLanguageMode(.v5)` pin lived only in `Package.swift`, so on the Swift 6.4
toolchain in use it silently stopped applying and the fork's mutable statics
(`Repo.overrideFolderNames`, `TtsCacheDirectory.overrideDirectory`) failed strict-concurrency
checks. The pin is now duplicated into both manifests. **Any future upgrade that touches either
manifest must keep them in sync.**

---

## 4. Fork patches carried forward

All re-applied and verified against the new APIs:

1. `Repo.overrideFolderNames` — app-controlled local folder names (`ModelNames.swift`, no conflict).
2. `ModelHub.loadModelsOnce` flat-layout fallback (was `DownloadUtils.loadModelsOnce`).
3. `VadManager.loadUnifiedModel` — skip the `Models/` level when given an external directory.
4. `TtsCacheDirectory.overrideDirectory` (was `TtsModels.overrideCacheDirectory`).
5. `G2PModel.resolveAssetURL` — flat-layout fallback for `g2p_vocab.json` and the G2P mlmodelcs.
6. `MultilingualG2PModel.modelsDirectory(base:)` — nested-vs-flat resolution.
7. PocketTTS `skipDownload` gate on `ensureModels` / `ensureMimiEncoder` / `ensureVoice`, plus the
   flat-layout fallbacks, merged with upstream's new `placement` / `computeUnits` parameters.
8. `KokoroAneResourceDownloader.ensureModels` — explicit-directory short-circuit.
9. `KokoroAneManager.initialize` — skip the G2P download when the override directory already holds
   the assets, merged with upstream's new best-effort lexicon prefetch.
10. `StreamingEouAsrManager` per-utterance EOU fix. Upstream refactored the debounce into the pure
    `evaluateEouDebounce`, which resets the *anchor* on new tokens but still never clears the sticky
    `eouDetected` outside `reset()`/`finish()` — so both halves of the fork patch are still needed.
    The token clear now also clears the two new token-aligned side arrays
    (`accumulatedTokenTimestampsMs`, `accumulatedRawTokenStrings`) to preserve their documented
    alignment invariant.
11. `Package.swift` Swift 5 language mode — **also added to `Package@swift-6.2.swift`** (see §3.7).
    The stale `exclude: ["Frameworks"]` was dropped; that directory no longer exists.

### New fork patches this upgrade

12. `KokoroAneResourceDownloader.ensureEnglishLexicon` — flat-layout check plus a hard offline gate
    when `TtsCacheDirectory.overrideDirectory` is set. Upstream added this best-effort fetch to
    `KokoroAneManager.initialize`; unpatched it reaches HuggingFace from the app sandbox on every
    English load.
13. `KokoroAneResourceDownloader.ensureVoicePack` — convert a **locally staged** repo-root
    `voices/<name>.json` before considering a network fetch, so all 54 English voices work offline
    from files `DownloadManagerCoreML` already downloads.
14. `PocketTtsResourceDownloader.ensurePackMimiEncoder` — flat-layout check and `skipDownload` gate.
    New upstream function on the non-English voice-cloning path; unpatched it ignores the caller's
    directory and downloads unconditionally.

---

## 5. Performance APIs available to `libs/audio/fluidaudio/` — audit result

Nothing in the app's wrappers is **broken** by the new APIs (every old signature the app calls is
still source-compatible). These are the opt-in wins, with what each would actually cost.

### 5.1 Parakeet v3 GPU encoder placement — **do not adopt unconditionally**

`AsrModels.load(from:configuration:version:encoderPrecision:encoderComputeUnits:progressHandler:)`
gained `encoderComputeUnits: MLComputeUnits?` (default `nil` → ANE). `.cpuAndGPU` is +~8% RTFx,
WER-neutral; upstream's own doc keeps ANE as the iOS default for power efficiency.

The app calls it at `FluidAudioASR.swift:238` and `:265` without the parameter. Adopting it would
have to be gated on `!forceCPU` — `ASRProtocol.swift:215` documents `forceCPU` as existing "for the
keyboard extension and in the background main app where GPU access is revoked". A blanket
`.cpuAndGPU` would break background and keyboard transcription outright. 8% for a battery cost and a
new conditional is a thin trade; **recommend leaving it.**

### 5.2 Parakeet `.int8V2` encoder — needs a download-config change first

`ParakeetEncoderPrecision` gained `.int8V2` (`Encoder_v2.mlmodelc`, 568 MB vs 425 MB). The app never
passes `encoderPrecision`, so it stays on `.int8`. Adopting means adding `Encoder_v2.mlmodelc` to the
`fa-parakeet-tdt-v3` components and shipping a bigger model for an unquantified accuracy delta.
**Not recommended without a measured comparison.**

### 5.3 EOU fused decoder — not reachable

`+7–9% RTFx` sounds attractive but there is no API for it. `StreamingEouAsrManager.loadModels(from:)`
enables it only when `FLUID_EOU_FUSED=1` **and** `decoder_joint_decision_fused.mlmodelc` sits in the
model directory. That filename is not in `ModelNames.ParakeetEOU.requiredModels`, so the standard
download never fetches it. Upstream keeps it off by default because the fused fp16 graph is not
bit-exact with the two-model reference. Adopting needs a config entry *and* an env var the app
cannot set for itself. **Not adoptable as shipped.**

### 5.4 PocketTTS `.ane` / `.aneState` — needs different model files

`PocketTtsManager.init` gained `placement:` and `computeUnits:`; the app passes neither, so it runs
`.gpu` + `.default`. Note the `~1.8× RTFx` headline belongs to the **v2.1 re-conversion the app
already gets**, not to switching placement. `.ane` loads different artifacts (`flowlm_step_ane`,
`cond_prefill_ane`) and `.aneState` a `pocket_state.mlmodelc` multifunction package (macOS 15+/iOS
18+ at runtime); both need new `components` entries. Upstream's own residency note is sobering: only
`flow_decoder_fused` actually lands on the ANE, and the earlier "flowlm 1.97× on ANE" claim *"did not
reproduce on-device."* The genuinely useful piece is `PocketTtsComputeUnits.avoidNeuralEngine` as a
**fallback** when a stage aborts on specific hardware, mirroring the ANE speaker's existing
`.default` → `.cpuOnly` retry. **Recommend the fallback only.**

### 5.5 Kokoro ANE compute units — already correct, by accident

`KokoroAneComputeUnits.default` became a **computed, OS-conditional** property:
`majorVersion >= 27 ? .aneTailCpu : .aneTailGpu`. The app passes `.default`
(`FluidAudioKokoroAneSpeaker.swift:394`), so it automatically picks up the OS-27 routing with no
change. Worth knowing that `.aneTailCpu` is documented as *"the lesser evil, not a safe one"* — see
§3.1.

### 5.6 Diarizer — two free wins, both small

- `OfflineDiarizerManager.process(audio:progressCallback:)` — the app calls
  `process(audio: samples)` at `FluidAudioDiarizer.swift:84` and `:119` with no progress callback, so
  a long meeting shows no progress. The parameter is defaulted, so wiring it is additive.
- `prepare()` / `cluster()` split — lets segmentation+embeddings be computed once and re-clustered
  with different settings. Only worth it if the app ever re-runs clustering, which it does not today.
- `exposeChunkEmbeddings` / `DiarizationResult.chunkEmbeddings` — off by default; no app use case yet.
- The LS-EEND memory-leak fix is internal and needs no app change.

### 5.7 `ModelHub.offlineMode` — **not** a drop-in for `skipDownload`

Process-wide, not per-call. The app deliberately passes `skipDownload: false` in one place
(`FluidAudioPocketTTSSpeaker.swift:432`, mimi encoder on first use for voice cloning); `offlineMode`
would block that too. It also throws `DownloadError.networkDisabled` / `.modelMissing` instead of
`PocketTTSError.modelNotFound`, so existing catches stop matching. It does add one guarantee the fork
flags lack — it blocks `loadModels`' retry-with-redownload fallback, so a corrupt-detected
`.mlmodelc` surfaces the load error instead of wiping the cache. **Worth considering later as a
belt-and-braces addition, not a replacement.**

---

## 6. Deliberately not adopted

| Upstream addition | Why not |
|---|---|
| `ModelHub.offlineMode` | Overlaps the fork's per-call `skipDownload` flags. Consolidating is a clean follow-up, not a merge requirement |
| ~~Paraformer-large (zh), SenseVoiceSmall~~ | **Adopted** 2026-09-18 — see §3.7 |
| Nemotron 3.5 Multilingual (40 locales), Parakeet Unified 0.6B, Canary-1B-v2 [beta] | New ASR backends; no app wrapper |
| Kokoro `ANE-ja` pack | Upstream ships it; the app neither downloads nor offers Japanese |
| Supertonic-3 (**31 languages**, 44.1 kHz, ~398 MB) | New TTS backend; no app wrapper. The commit title says "10-lang results" — that is the benchmark subset, not the model's coverage |
| LuxTTS, NeuTTS-2E [beta], Chatterbox (Multilingual + Nano), Inflect v2 [beta], StyleTTS2 | New TTS backends; no app wrapper |
| FSMN-VAD [beta], CAM++ speaker embeddings [beta] | New VAD / speaker backends; no app wrapper |
| PocketTTS `.ane` / `.aneState` placements, per-stage compute units | Available and plumbed through the fork's patches; the app still passes the `.gpu` default |
| Parakeet v3 GPU encoder placement, int8 Encoder_v2, EOU fused decoder | Opt-in perf flags the app does not set |
