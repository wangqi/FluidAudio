# FluidAudio Upgrade Notes: tag-20260918 → tag-20260928

**31 commits, 2026-09-19 → 2026-09-26** (8 `feat`, 11 `fix`, 1 `perf`, 4 `docs`, plus 3 untyped
feature commits and 4 README/showcase edits). Upstream versions **v0.17.0 → v0.17.4** (podspec).
149 files changed, +15,816 / −364; 71 of them under `Sources/` (+7,798 / −315). Most of the
added lines are new backends the app does not wrap (Nemotron 3 diarization, LocalVQE, CUA-S1,
Spanish/French Kokoro G2P, LuxTTS chunking).

This is a small merge next to tag-20260918, but it contained **one silent ship-blocker**:
upstream renamed a Kokoro stage bundle, and our HuggingFace mirror did not have the new file.
**Resolved 2026-09-28** (mirror updated, test tier synced); see §3.1 for what is still pending
at release.

All 14 fork patches carried through the merge unchanged. The per-file fork delta against
upstream `main` is identical in size to the tag-20260918 delta. One patch needed a follow-up
fix; see §4.

---

## 1. Upstream changes, by area

### 1.1 Kokoro ANE (the app's FluidAudio Kokoro engine)

| Commit | What it does | Reaches the app? |
|---|---|---|
| `fix(tts/kokoro-ane): quiet onset on long utterances — use fp32 KokoroProsody_v2 (#947) (#963)` | The shipped fp16 prosody stage miscomputed F0/energy over the first 1–3 s once an utterance passed ~10 s of audio, so the **opening words came out 12–15 dB quiet**. Not iOS-specific; every variant affected. The stage is now `KokoroProsody_v2.mlmodelc` (fp32 compute, same fp16 I/O). It was **renamed, not overwritten**. Reporter's sentence: onset −38.2 dB → −26.5 dB, ASR WER 7.7% → 3.8% | **Yes, but only after the mirror is updated.** See §3.1 |
| `fix(tts/kokoro-ane): restore long-text chunking in synthesizeDetailed(text:) (#940) (#965)` | #790 had dropped the phoneme chunker, so text past 510 phonemes threw `phonemeSequenceTooLong`. It is chunked again, after normalization and G2P, and the cap is now counted in **Unicode scalars**. `synthesize(text:)` goes through `synthesizeDetailed`, so both are fixed | **Yes, directly.** See §2.1 |
| `feat(tts/kokoro-ane): Spanish and French variants (#950)` | `KokoroAneVariant.spanish` / `.french` share the `ANE/` bundle, with new frontends: `SpanishG2P` (spelling rules plus a 49k-word `es_lexicon_cache.json`) and `FrenchG2P` (244k-word `fr_lexicon_cache.json`, CharsiuG2P fallback, elision and liaison). Phoneme error vs espeak-ng on FLEURS: es 0.48%, fr 1.27% | Not adopted. See §5.2 |
| (same commit) Vocoder tail-slack fix | On OS 27, the BNNS kernels that Core ML runs for CPU-routed ops **read a few bytes past the end of their input**. When a buffer's byte size is a whole number of pages, that read lands on an unmapped page and **segfaults in libBNNS**; otherwise it produced full-scale noise. Hit at T ≡ 8 mod 24 in 416–584. Every chain input now carries a 16 KB zeroed tail | **Yes.** Upstream says it is *"possibly the same class as #889 (iOS 27); not verified there"*. See §3.2 |
| (same commit) `KokoroAneVocab` encodes by Unicode scalar | French nasal vowels (`ɑ̃` = two scalars) were being dropped | No: the app runs English and Mandarin only |
| `feat(tts/kokoro-ane): expose normalizedText + phonemes on synthesizeDetailed (#944)` | For callers that align display words to per-token durations | No: the app does not highlight words for this engine |

### 1.2 ASR

| Commit | What it does | Reaches the app? |
|---|---|---|
| `perf(asr): bulk-fill MLMultiArray resets and copies (#941)` | `resetData(to:)` became one `memset` and `copyData(from:)` one `memcpy`, replacing a per-element `NSNumber` loop. `MLArrayCache.returnArray` also stopped clearing arrays that every consumer overwrites anyway. `AsrManager.transcribe` on an 11 s clip went **71.1 ms → 48.3 ms** (M3 Pro, release, median of 20), with identical transcripts | **Yes, free.** Parakeet v3 and Japanese |
| `Load local Orukeet-compatible ASR bundles without repository fallback (#928)` | New `AsrModels.loadLocal(from:version:…)`: loads from the exact directory, **no repository resolution, no download** | **Adopted.** See §5.1 |
| `feat(asr): AsrModelVersion.ultra — Parakeet Ultra (#956)` | Post-trained v3 with the same contract, iOS 17+. test-clean WER 2.27% → **2.13%**, test-other 4.12% → **3.81%**, FLEURS 24-language mean 14.81% → **11.67%** (won all 24). One int8 encoder (595 MB). Upstream now *recommends* it for new integrations, but v3 stays the library default | Not adopted. See §6 |
| `feat(asr): AsrModelVersion.redux — Parakeet Redux (#955)` | Ternary (2-bit) re-training of v3: **183 MB encoder** vs 445 MB, **iOS 18+ only**. Slightly worse on English (2.71% / 5.12%), better on FLEURS (13.06%), ~35% slower (83.9× vs 128.6× RTFx). First ANE load compiles ~7 min on an M-series Mac | Not adopted. See §6 |
| `feat(diarizer): … SenseVoiceManager.transcribeDetailed` (inside #883) | Returns the model's leading tags: detected language (`zh`/`en`/`yue`/`ja`/`ko`/`nospeech`), emotion, audio event | **Adopted.** See §5.3 |
| `fix(vocab): make alignBaseWordsToUTF8Ranges iterative (#961) (#962)` | Custom-vocabulary rescoring recursed once per word and SIGBUSed at ~1–2k words inside a Swift Task. Now an explicit stack: 16k words in 7 ms | No: the app does not use vocabulary boosting |

### 1.3 PocketTTS

`fix(tts/pocket): keep cache-safe long sentences whole (#938)`. The 50-token chunk size is now a
grouping preference, not a hard sentence limit. A longer sentence is kept whole when the chosen
voice's KV offset plus a speech-duration budget still fits the 512-position cache. Generation
loops are bounded at that boundary, and every cache write is validated. **Reaches the app**:
long sentences are no longer cut mid-clause, and a cache overrun now fails a bounds check
instead of reading past the cache. There are no model or file-name changes, so there is no
catalogue impact.

### 1.4 Diarization

- `feat(diarizer): Nemotron 3 Diarization support (8-speaker streaming Sortformer) (#883)`: an
  NVIDIA general-access checkpoint with 8 speakers, 10 ms output resolution, and a streaming API
  (`appendAudio` / `processBufferedAudio` / `finishStream`, bit-exact with `processComplete`). A
  split-graph mode gives 100% ANE residency. AMI 16-meeting DER 9.36–9.75 depending on preset.
- `fix(diarizer/nemotron3): type output backings from the model description, retry if rejected (#952)`
  and `feat(diarizer/nemotron3): load monolithic presets from monolithic/v2 (M3 ANE compile fix) (#960)`:
  follow-ups for the M3 (h15g) `ANECCompile()` failure.
- `Pin diarization artifacts to immutable revision (#927) (#939)`: `Repo.diarizer` now downloads
  from commit `df2625ac…`, not `main`. New `ModelRegistry.revisionOverrides` covers mirrors, and
  a `.fluidaudio-revision` marker invalidates the cache when the pin moves. See §3.4.

### 1.5 New, outside the app's scope

- `feat(enhancement): LocalVQE AEC with safe streaming and benchmark validation (beta) (#930)`:
  acoustic echo cancellation, noise suppression and dereverberation for 16 kHz speech.
  `LocalVqeManager` handles whole clips; `LocalVqeStream` takes arbitrary buffer sizes. On
  upstream's exploratory ASR study, leakage fell from 34.0% to 0.95%. **Live capture/playback
  on target devices is explicitly unvalidated upstream.**
- `Add CUA-S1-FORMS Core ML scoring and benchmarks (#936)`: an on-device decision scorer that
  picks one of 2–32 UI actions. Not audio.
- `fix(tts/luxtts): remove spurious mid-phrase pauses and chunk long text (#937) (#942)`: LuxTTS
  is not wrapped by the app.
- `feat(logger): AppLogger.minimumLevel + mirrorsToConsole (#958) (#959)`: see §5.4.
- `fix(build): bump NemoTextProcessing to v0.3.1 for Mac Catalyst (#949)`: adds a Catalyst slice.
  **Inert here**: the fork disables the `NemoTextProcessing` trait (tag-20260918 §3.2, the
  libgit2 `module.modulemap` collision), and this bump does not change that.

---

## 2. What users get from this merge

### 2.1 Kokoro Text to Audio stops failing on long chunks

This is the most concrete user-facing fix, and it was already broken before this merge.
`TTSConvModels.defaultChunkMaxChars(for: .fluidAudio)` is **600 characters**, and
`FluidAudioKokoroAneSpeaker.generateAudioFile` passes the whole chunk to one `synthesize(text:)`
call. English runs at about 1.02 phonemes per character (upstream's 916-char paragraph was 936
phonemes), so a full 600-character chunk is ~610 phonemes. That is **over the 510 cap, so it
threw `phonemeSequenceTooLong`**. Upstream's restored chunking removes the failure with no app
change. Interactive read-aloud (`speakText`) was mostly safe: its own `splitIntoChunks` keeps
chunks to at most ~500 characters, which sits right at the cap, so an unusually phoneme-dense
paragraph could still have thrown there. It no longer can.

### 2.2 Kokoro's opening words are no longer quiet

This fix applies once the mirror carries `KokoroProsody_v2.mlmodelc` (§3.1). On any utterance
longer than ~10 s of audio, the first words used to come out 12–15 dB quieter than the rest.

### 2.3 Faster Parakeet transcription

About 20 ms is saved per `transcribe` call on an 11 s clip (~30%). The saving comes from work
after the models finish, so it applies equally on the Neural Engine and the CPU.

### 2.4 SenseVoice reports the language it heard

See §5.3.

---

## 3. Risk assessment

### 3.1 RESOLVED (was ship-blocker) — `KokoroProsody_v2.mlmodelc` was not in our mirror

`ModelNames.KokoroAne.prosody` changed from `KokoroProsody.mlmodelc` to
`KokoroProsody_v2.mlmodelc`. Every catalogue row is served from our own mirror,
`flyingfishinwater/fluidaudios`. On 2026-09-28 the live tree listed:

| Path | Has `KokoroProsody_v2.mlmodelc`? |
|---|---|
| `FluidInference/kokoro-82m-coreml/ANE/` (upstream) | yes |
| `FluidInference/kokoro-82m-coreml/ANE-zh/` (upstream) | yes |
| `flyingfishinwater/fluidaudios/kokoro-82m-coreml/ANE/` (**ours**) | **no**: only `KokoroProsody.mlmodelc` |
| `flyingfishinwater/fluidaudios/kokoro-82m-coreml/ANE-zh/` (**ours**) | **no** |

**Consequence if shipped as-is:** FluidAudio Kokoro produces no audio at all, on every install,
new or existing, in English and Mandarin. The fork's explicit-directory patch (§4, patch 8) is
designed not to reach HuggingFace, so the model store's per-file guard throws, and the
`.cpuOnly` retry in `FluidAudioKokoroAneSpeaker.loadManagerIfNeeded` throws the same way.
There is no compile error and no failing test: the catalogue's component is the whole `ANE`
directory, so `FluidAudioCatalogueSchemaTests.assertCovers(…, scope: "ANE")` is satisfied by
whatever the folder contains. This is the same failure class as tag-20260918's `KokoroNoise_v2`
and PocketTTS renames.

**Done in this branch:**

- `fa-kokoro-82m` `components.revision` bumped **1 → 2** in all four tiers
  (`helper/fluidaudio_model{,_test,_audit}.json`, `data/fluidaudio_model.json`, still
  byte-identical). This makes existing installs report "not downloaded" and fetch the folder
  again. **Not synced to S3.**
- New gate `FluidAudioCatalogueSchemaTests.testKokoroAneStageNamesMatchTheMirroredRevision`
  freezes `ModelNames.KokoroAne.requiredCoreMLModels` to the seven mirrored names and pins
  revision 2. The next upstream stage rename fails that test by name.

**Resolved 2026-09-28**, in this order:

1. Uploaded both bundles to the mirror, exactly as upstream publishes them (four files each; v2,
   like upstream, ships no `metadata.json`). Mirror commit
   [`3dbab147`](https://huggingface.co/flyingfishinwater/fluidaudios/commit/3dbab14721f1c46a903fe1ac0c4070e2bf5b230a).
   The old `KokoroProsody.mlmodelc` is kept in both folders for older app versions. The
   `weight.bin` SHA-256 fetched back from the mirror matches upstream: `ANE` `70eea4cd…`,
   `ANE-zh` `9835a7d1…`.
2. Re-listed the mirror tree: all eight files are present.
3. Synced the test tier: `./sync_models_json.sh fluidaudio_model test`. **Audit and prod are not
   synced**; they go out with the release.

The procedure, for the record:

1. Upload `KokoroProsody_v2.mlmodelc` from upstream into the mirror under **both**
   `kokoro-82m-coreml/ANE/` and `kokoro-82m-coreml/ANE-zh/`. Use the Python API, as
   `helper/docs/audio-fluidaudio.md` §5 describes:
   ```python
   from huggingface_hub import snapshot_download, HfApi
   for v in ["ANE", "ANE-zh"]:
       d = snapshot_download(repo_id="FluidInference/kokoro-82m-coreml",
                             allow_patterns=[f"{v}/KokoroProsody_v2.mlmodelc/**"],
                             local_dir="/Volumes/ssd2t/fluidaudio/_prosody_v2")
   HfApi().upload_folder(repo_id="flyingfishinwater/fluidaudios",
                         folder_path="/Volumes/ssd2t/fluidaudio/_prosody_v2",
                         path_in_repo="kokoro-82m-coreml",
                         allow_patterns=["ANE/KokoroProsody_v2.mlmodelc/**",
                                         "ANE-zh/KokoroProsody_v2.mlmodelc/**"])
   ```
2. Re-list the mirror tree and confirm both bundles are present.
3. Only then run `cd helper && ./sync_models_json.sh fluidaudio_model test`.

**Cost of the revision bump:** the catalogue is remote and shared across app versions. Once the
audit and prod tiers carry revision 2, **every device with FluidAudio Kokoro re-downloads the
~1.4 GB pack, including devices still on older app versions.** Those devices keep working: they
load `KokoroProsody.mlmodelc`, which stays in the mirror. The download is wasted for them,
though. The alternative is worse, because without the bump the new app version cannot load
Kokoro at all on an existing install. **Keep the old `KokoroProsody.mlmodelc` in the mirror**:
older app versions still need it.

### 3.2 HIGH, reduced — iOS 27 Kokoro crash advisory is still in force

`KokoroAneManager.osAdvisory` is unchanged. It still flags every non-macOS OS ≥ 26.4, and on
iOS 27 it still says *"no Core ML route for Kokoro ANE is known to be safe"* (#889). This merge
**lowers but does not remove** the risk. The vocoder tail-slack fix (§1.1) addresses a
confirmed BNNS past-the-end read on OS 27, which crashed on macOS 27. Upstream suspects #889 is
the same class but has not verified it on iOS. The product decision recorded in tag-20260918
§3.1 (keep, cap, or hide FluidAudio Kokoro on iOS 27) still stands. This merge is evidence for
revisiting it, not a resolution.

### 3.3 MEDIUM — `AsrModels.loadLocal` is stricter than `load`

§5.1 moves Parakeet v3, v2 and Japanese onto `loadLocal`. What changes:

- A missing file is now a local `AsrModelsError.modelNotFound`. Before, `ModelHub` fetched it
  from **upstream** `FluidInference/*`, not our mirror, which bypassed the pinning the mirror
  exists for.
- A model that fails to load is no longer purged and re-downloaded
  (`ModelCache.purgeCorruptedCache` in `ModelHub.loadWithRecovery`). A transient ANE compile
  failure used to cost the user the whole ~620 MB folder.
- `loadLocal` rejects a vocabulary missing any id below the blank id. Checked against the
  staged files in `/Volumes/ssd2t/fluidaudio`: v3 `parakeet_vocab.json` has 8192 contiguous
  ids (blank 8192) and Japanese `vocab.json` has 3072 (blank 3072). Both pass.
- Compute units are identical: preprocessor `.cpuOnly`, everything else from
  `defaultConfiguration()` (`.cpuAndNeuralEngine`).

The risk is a file-name assumption the harness has not exercised. It is gated by
`FluidAudioCatalogueRegressionTests`, whose `probeParakeet` now calls the same `loadLocal`.

### 3.4 LOW, dormant — diarizer revision pin can wipe an app-staged folder

For a pinned repo (`Repo.diarizer`), `ModelHub` treats a cache with no `.fluidaudio-revision`
marker as stale. It marks every file incomplete and calls `ModelCache.prepareForDownload`,
which **deletes the folder** and re-downloads from upstream. `DownloadManagerCoreML` never
writes that marker. So the offline diarizer (`fa-speaker-diar`) would, on first load, delete a
pack the app staged and fetch it again from `FluidInference/speaker-diarization-coreml`.

**Dormant today:** the three diarizer rows exist only in the orphaned
`helper/fluidaudio_model_global.json`, so no shipping tier can download them. It must be
handled before any diarizer row reaches a live tier. The simplest fix is
`ModelRegistry.revisionOverrides["FluidInference/speaker-diarization-coreml"] = "main"` at
startup, the escape hatch upstream added for mirrors.

### 3.5 LOW — PocketTTS phrasing changes

Sentences over 50 tokens are now synthesized whole where the cache allows. Output for the same
text will differ from tag-20260918 at those sentence boundaries. Upstream frames this as a fix
(no more mid-clause breaks), but anything that compares PocketTTS audio across app versions will
see a difference.

### 3.6 NONE — everything else

No other `ModelNames` constant used by a catalogue row changed; §3.1 is the only rename. No app
code switches exhaustively over `KokoroAneVariant` or `AsrModelVersion`, both of which gained
cases, so the new cases do not break compilation. `Package.swift` and `Package@swift-6.2.swift`
still carry the fork's `.swiftLanguageMode(.v5)` and the disabled `NemoTextProcessing` trait.

---

## 4. Fork patches

All 14 from tag-20260918 are carried unchanged: `Repo.overrideFolderNames`, the `ModelHub` and
`G2PModel` / `MultilingualG2PModel` flat-layout fallbacks, the `VadManager` external-directory
patch, `TtsCacheDirectory.overrideDirectory`, the PocketTTS `skipDownload` gates, the Kokoro ANE
explicit-directory short-circuit, the G2P-skip, lexicon and voice-pack offline gates, the
`StreamingEouAsrManager` per-utterance EOU fix, and Swift 5 language mode in both manifests.

**Fixed in this merge**, patch 8 (`KokoroAneResourceDownloader.ensureModels` explicit-directory
short-circuit). It chose the required set with `variant == .english ? requiredModels :
requiredModelsZh`. Upstream grew `KokoroAneVariant` to five cases, so `.spanish`, `.french` and
`.japanese` were held to Mandarin's `g2pw/` and `voices/zf_001.bin`. It now switches per
variant, matching upstream's own switch below it. The only effect was a misleading warning
(the patch never throws), but it would have misreported any Spanish or French adoption.

**Not yet patched: two new upstream network paths**, both reachable only from the Spanish and
French variants the app does not use:

- `KokoroAneResourceDownloader.ensureLexiconCache(_:directory:)` (`es_`/`fr_lexicon_cache.json`)
  resolves `<TtsCacheDirectory>/Models/kokoro/<file>` and downloads from upstream when it is
  absent. It has no flat-layout check and no offline gate, unlike its English sibling
  `ensureEnglishLexicon` (patch 12).
- `KokoroAneResourceDownloader.ensureMultilingualG2PAssets` calls `ModelHub.download` for the
  CharsiuG2P pair.

Both need the patch-12 treatment before Spanish or French is wired (§5.2).

---

## 5. `libs/audio/fluidaudio/` audit

### 5.1 ADOPTED — `AsrModels.loadLocal` for Parakeet v3 / v2 / Japanese

`FluidAudioASR.loadManagerIfNeeded` used to write the process-global
`Repo.overrideFolderNames[.parakeetV3 | .parakeetV2 | .parakeetJa] = cacheFolder` and then call
`AsrModels.load(from:)`. That relied on `load` doing `directory.deletingLastPathComponent()` and
re-appending the overridden `folderName`. It now calls
`AsrModels.loadLocal(from: modelURL, version:)`, which is the upstream API made for exactly this
case. Trade-offs are in §3.3. `FluidAudioCatalogueRegressionTests.probeParakeet` was changed to
match, so the regression harness still exercises the app's own load path.

With this change the app no longer writes `Repo.overrideFolderNames` anywhere. The fork patch
that adds it stays, because the regression harness and `FluidAudioVAD`'s open issue
(tag-20260918 §3.12, which suggested an override for `.vad`) may still want it.

### 5.2 NOT ADOPTED, worth doing — Kokoro Spanish and French

The `fa-kokoro-82m` catalogue already lists `ef_dora`, `em_alex`, `em_santa` and `ff_siwis` and
advertises `es`/`fr` in `language_list`. Today those voices run on the **English** variant, so
Spanish or French text is spoken through English G2P with a Spanish or French timbre.
Upstream's new variants fix that **without a new speaker class**: `FluidAudioKokoroAneSpeaker`
would map the `ef_`/`em_` prefixes to `.spanish` and `ff_` to `.french` when constructing
`KokoroAneManager`. Prerequisites:

1. Mirror `es_lexicon_cache.json` (3.9 MB) and `fr_lexicon_cache.json` (13.6 MB) to the
   `kokoro-82m-coreml/` root. Neither is in our mirror today. `MultilingualG2PEncoder/Decoder.mlmodelc`
   already are.
2. Add the flat-layout + offline gates from §4 to `ensureLexiconCache` and
   `ensureMultilingualG2PAssets`.
3. Reload the manager when the voice crosses a language, since a manager is bound to one
   variant. The seven stage bundles are the same `ANE/` files, so this costs a reload, not a
   download.
4. Bump `fa-kokoro-82m` `components.revision` again if the lexicons join the default download.

### 5.3 ADOPTED — SenseVoice detected language

The SenseVoice branch now calls `transcribeDetailed(audio:)` and returns the model's language
tag in `ASRResult.language`, falling back to the caller's hint. SenseVoice takes no language
input, so the tag is what was actually recognised. That matches how `WhisperKitASR` fills the
same field. A `nospeech` tag is treated as no language. The text is identical to
`transcribe(audio:)`, which is now a wrapper over `transcribeDetailed`.

### 5.4 RECOMMENDED, not applied — `FluidAudio.AppLogger.minimumLevel`

In Debug builds FluidAudio mirrors every log level to stderr, and some ASR debug lines include
recognised words. `FluidAudio.AppLogger.minimumLevel = .info` at startup keeps transcript text
out of both `os_log` and the console. Release builds already go to `os_log` only, so the benefit
is limited to developer builds. The type is qualified as `FluidAudio.AppLogger` because the app
has its own `AppLogger`.

### 5.5 Free — no app change needed

- Parakeet `memset`/`memcpy` speedup (§2.3).
- Kokoro long-text chunking (§2.1). `FluidAudioKokoroAneSpeaker.splitIntoChunks` stays: it is
  what gives `speakText` its per-chunk playback and `onChunkStarted` callbacks, not a length
  workaround any more.
- Kokoro vocoder tail slack (§3.2).
- PocketTTS long-sentence handling (§1.3).

### 5.6 Still standing from tag-20260918 §5

The Parakeet v3 GPU encoder, the `.int8V2` encoder, the EOU fused decoder, PocketTTS `.ane`
placements, and `ModelHub.offlineMode` are unchanged by this merge. The recommendations there
still apply.

---

## 6. New models that would need new app code

**None is required for this upgrade.** If you want any of these, this is the work:

| Model | New class? | What it takes |
|---|---|---|
| **Parakeet Ultra** (`AsrModelVersion.ultra`) | **No**, a branch in `FluidAudioASR` | Mirror `FluidInference/parakeet-ultra-coreml`, add an `fa-parakeet-ultra` row + `repoMap` entry, and dispatch `loadLocal(from:version: .ultra)` (the §5.1 change makes this one line). Same size class as v3 (~620 MB). **Best candidate**: more accurate than v3 on every set upstream measured, at ~v3 speed, and upstream now recommends it |
| **Parakeet Redux** (`.redux`) | **No**, same branch | Same wiring as Ultra. A smaller download (183 MB encoder) but **iOS 18+ only**, less accurate on English, ~35% slower, and a ~7 min first ANE compile on a Mac. It only makes sense as a "small download" option |
| **Nemotron 3 Diarization** | **Yes, a new diarizer case**: a fourth `DiarizerType` in `FluidAudioDiarizer`, around `Nemotron3Diarizer` + `Nemotron3Models` | Different API shape from LS-EEND / Sortformer (`Nemotron3ChunkResult`, `Nemotron3Diarizer.segments(…)`), 8 speakers vs Sortformer's 4, 10 ms resolution. Needs catalogue rows too, and **no FluidAudio diarizer row is in any live tier today** (§3.4) |
| **LocalVQE** (echo cancellation / noise suppression) | **Yes, a new component type**, not a speaker, ASR, VAD or diarizer | A pre-ASR enhancement stage (`LocalVqeStream`) for voice chat, where the app's own TTS leaks into the mic. **Beta**, 16 kHz only, CPU default, live-device behaviour unvalidated upstream |
| **Kokoro Spanish / French** | **No**, extends `FluidAudioKokoroAneSpeaker` | §5.2 |

---

## 7. Deliberately not adopted

| Upstream addition | Why not |
|---|---|
| CUA-S1-FORMS decision scorer | Not audio; the app's decision engine is Laya |
| LuxTTS chunking and pause fixes | LuxTTS has no app wrapper |
| Custom-vocabulary SIGBUS fix | The app does not use vocabulary boosting |
| `normalizedText` / `phonemes` on `KokoroAneSynthesisResult` | No word-highlighting consumer for this engine |
| NemoTextProcessing v0.3.1 | The trait stays disabled in the fork (tag-20260918 §3.2) |
| Diarizer `revisionOverrides` | Nothing to override until a diarizer row ships (§3.4) |

---

## 8. Verification (2026-09-28)

| Check | Result |
|---|---|
| iOS Simulator test build (`AIAssistantFunctionTests`) | build succeeded |
| macOS test host (`AIAssistantMacUnitTests`, built by the harness) | build succeeded |
| `FluidAudioCatalogueSchemaTests` (incl. the new `testKokoroAneStageNamesMatchTheMirroredRevision`) | 10/10 pass |
| `FluidAudioASRDispatchTests` | 11/11 pass |
| `FluidAudioKokoroAneSpeakerTests` | 4/4 pass |
| `run_fluidaudio_tests.py --type stt` on real weights (`/Volumes/ssd2t/fluidaudio`) | **5/5 PASS**: Parakeet v3 via `loadLocal` similarity 1.0; Japanese via `loadLocal` loads and decodes (no oracle); SenseVoice CER 0.10; Paraformer CER 0.133; EOU similarity 0.909 |
| `run_fluidaudio_tests.py --only kokoro`, before the mirror upload | **2/2 FAIL, as predicted by §3.1**: `KokoroAneError: KokoroAne model 'KokoroProsody_v2.mlmodelc' not loaded`, for both `ANE` and `ANE-zh` |
| Same, after the upload, with `KokoroProsody_v2.mlmodelc` re-staged **from our mirror** | `ANE` (English) **PASS**, 3.375 s of audio, rms 0.047. `ANE-zh` fails with `You don't have permission to save the file "g2p" in the folder "ANE-zh"`. That is the known read-only-stage limit documented in `helper/docs/audio-fluidaudio.md` (Mandarin G2P fetches its text assets lazily), identical to both tag-20260918 runs, and it now fails *after* the stage models load instead of at the missing prosody bundle |

Reports: `/Volumes/ssd2t/modeltests/reports/fluidaudio-20260928-143244.md` (stt),
`fluidaudio-20260928-143336.md` (kokoro, before the upload), `fluidaudio-20260928-143955.md`
(kokoro, after).
