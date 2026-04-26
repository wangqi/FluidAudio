# FluidAudio What's New: tag-20260419 → tag-20260425

## Summary

This upgrade includes one new ASR engine (Cohere Transcribe), critical correctness fixes for offline diarization, a major concurrency safety fix for the diarizer on iOS 26.4, a bug fix for Parakeet v3 emitting Cyrillic on Latin-script languages, and a Kokoro abbreviation normalization fix.

---

## New Features

### Cohere Transcribe ASR (#487, #537)

A new multilingual ASR engine based on Cohere's INT8 encoder + FP16 cache-external decoder hybrid (`CoherePipeline`). Supports 14 languages:

- English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Greek, Arabic, Japanese, Chinese (Simplified), Korean, Vietnamese

**Key design:**
- `CohereMelSpectrogram`: 128-mel front-end matching Cohere's reference model (preemph, Slaney mel, CMVN)
- `CoherePipeline`: encoder + cache-external decoder; K/V cache allocated host-side (no CoreML State API); SentencePiece byte-fallback detokenization for CJK
- `CohereAsrConfig`: per-language prompt sequences, 35 s audio window, 108-token decoder cache window
- Model hosted at `FluidInference/cohere-transcribe-03-2026-coreml/q8`

**ANE-friendly decoder v2 (#537):**
- New `cohere_decoder_cache_external_v2` with fully static attention mask `[1, 1, 1, 108]` — unblocks ANE dispatch
- v1 (dynamic `RangeDim(1, 108)`) falls back to CPU/GPU; v2 lands fully on ANE
- Measured median decoder time: v2 = **2.58 s** vs v1 = **4.13 s** (~1.6x faster) on identical audio
- `CoherePipeline.loadModels`: prefers v2, falls back to v1 automatically

**iOS relevance:** Full iOS 17+ support. ANE speedup is significant on A-series chips.

---

## Bug Fixes

### Offline Diarization: Three Critical Correctness Bugs Fixed (#523)

The offline diarization pipeline was producing nearly single-speaker output due to three compounding bugs:

1. **`vDSP_mtrans` dimension swap** — `frameCount` and `speakerCount` arguments were reversed, corrupting the transposed speaker masks to be nearly identical across all speakers.
2. **Missing activity ratio filter** — pyannote reference filters speakers with <20% clean activation; this code only filtered completely silent speakers, allowing junk embeddings through to clustering.
3. **Soft masks vs binary masks** — reference uses argmax on powerset logits (binary 0/1); this code used soft probabilistic masks via matrix-vector multiplication, producing blurred activations.

**After fix:** 97% F1 score on a 467-second 3-speaker audio file at 120x real-time.

**iOS impact:** Offline diarization was effectively broken before this fix. High risk if any workflows depend on `OfflineDiarizerManager` for multi-speaker attribution.

### Diarizer: SpeakerManager Actor Conversion for iOS 26.4 (#528, #539)

`SpeakerManager` converted from class + `DispatchQueue.sync(flags: .barrier)` to `actor`. `Speaker` changed from `final class` to `struct` (Sendable value type).

**Root cause:** `DispatchQueue.sync` in async context triggered `unsafeForcedSync` warnings under Swift 6 strict concurrency, and concurrent COW mutations on `[Float]` embedding buffers caused `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block` on iOS 26.4.

**Changes:**
- `SpeakerManager`: class → `actor` (all mutating methods now `async`)
- `Speaker`: `final class` → `struct`
- `DiarizerManager`: `initializeKnownSpeakers()` and `extractSpeakerEmbedding()` methods now `async`

**iOS impact:** Heap corruption was reproducible on iOS 26.4 when `DiarizerModels.download()` and `SpeakerManager.extractSpeakerEmbedding` were called from async contexts. This is a critical stability fix for iOS 26.4+.

### Parakeet TDT v3: Cyrillic Output on Latin-Script Languages (#512, #515)

Parakeet TDT v3 transcribed short Polish utterances as Cyrillic (e.g. "Wpisz Google kropka com" → "Впиш Гугл к ком.") because the joint decoder's top-1 pick drifts to Cyrillic tokens under low acoustic confidence.

**Fix:** New opt-in `Language` enum + `TokenLanguageFilter` — pass `language: .polish` (or any other declared-script language) to `transcribe()`. The decoder rejects top-1 Cyrillic tokens and walks top-K to the highest-probability candidate matching the target script. Requires `JointDecisionv3.mlmodelc` (auto-downloaded with v3 model).

**Languages with declared scripts:**
- Latin: English, Spanish, French, German, Italian, Portuguese, Romanian, Polish, Czech, Slovak, Slovenian, Croatian, Bosnian
- Cyrillic: Russian, Ukrainian, Belarusian, Bulgarian, Serbian

**Opt-in: `language:` defaults to `nil` — zero behavior change for existing callers.**

### Kokoro: Abbreviation Handling Fixed (#538)

Two bugs in Kokoro's G2P abbreviation normalizer:
- Abbreviations were not sorted by longest key first, causing "etc." to sometimes match "etc" (shorter key) and leave a stray ".".
- Trailing `\b` word-boundary fails when the abbreviation ends in a non-word character (e.g. "Dr." followed by a space).

Both bugs are fixed upstream in `FluidAudio`'s Kokoro G2P pipeline. No changes needed in the app wrapper.

### EOU Sticky Flag and Token Accumulator (committed by wangqi, a5ca8e0)

`StreamingEouAsrManager`: fixed the EOU sticky flag and token accumulator so multi-utterance streaming correctly emits per-utterance events instead of accumulating all utterances into one.

---

## Upgrade Risk Assessment

| Area | Risk | Notes |
|---|---|---|
| Offline diarization correctness | **HIGH** (previously broken) | Results will change after fix; multi-speaker output now correct |
| iOS 26.4 diarizer stability | **HIGH** (crash/heap corruption fixed) | Must upgrade before shipping on iOS 26.4 |
| Cohere Transcribe ASR | **LOW** (new engine, no API changes) | New models needed; existing Parakeet unaffected |
| Parakeet v3 Cyrillic fix | **LOW** (opt-in only) | No behavior change unless `language:` is passed |
| Kokoro abbreviation fix | **LOW** (upstream G2P fix) | May change phoneme output for abbreviation-heavy text; test TTS output |
| SpeakerManager actor API | **MEDIUM** | `initializeKnownSpeakers()` / `extractSpeakerEmbedding()` now async; any direct caller must `await`. Our wrappers use `OfflineDiarizerManager` / `LSEENDDiarizer` / `SortformerDiarizer` which abstract this — no change needed in app code. |

---

## App Integration Notes

### FluidAudioDiarizer
No code changes required. The actor conversion and offline pipeline fixes are internal to FluidAudio. Offline diarization results will be substantially more accurate (was single-speaker; now 97% F1 F-measure).

### FluidAudioASR
The new `language:` parameter on `AsrManager.transcribe()` is opt-in and defaults to `nil`. Existing code compiles unchanged. Callers transcribing Polish or other Latin-script languages that may drift to Cyrillic should pass `language: .polish` etc. to activate the script filter.

### FluidAudioKokoroSpeaker
No code changes required. The abbreviation fix is in FluidAudio's upstream G2P pipeline.

### New Model: Cohere Transcribe
A new `FluidAudioCohereASR` (or additional modelId in `FluidAudioASR`) would be needed to expose Cohere Transcribe. Not strictly required for this upgrade — existing functionality is unaffected. See "New Model Recommendation" below.

---

## New Model Recommendation

**Cohere Transcribe** is a strong multilingual ASR option not yet exposed in the app. It covers 14 languages with ANE-accelerated inference on A-series chips. Integration would require:
- New model download entry in `models_*.json` pointing to `FluidInference/cohere-transcribe-03-2026-coreml/q8`
- New class `FluidAudioCohereASR` wrapping `CoherePipeline` (separate from `FluidAudioASR` due to different pipeline: encoder + decoder directories, no `TdtDecoderState`)
- Add to `fluidAudioSupportedModels` in `LocalModelAboutView`

This is optional and can be done in a follow-up PR.
