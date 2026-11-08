#if TTS
@preconcurrency import CoreML
import Foundation

// Doc line for the caller-supplied conditioning entry points // wangqi modified 2026-11-07
/// Text-to-speech with the small Pocket TTS models (18m / 7m / 2m): the FlowLM
/// on the Neural Engine through one MLState multifunction package, the Mimi
/// decoder in fp32 on the CPU by default (`mimiPrecision`, see
/// `PocketTtsSmallMimiPrecision` for the fused all-ANE alternative).
///
/// ```swift
/// let tts = PocketTtsSmallManager(model: .english7m)
/// try await tts.initialize()
/// let wav = try await tts.synthesize(text: "Hello, world!")
/// ```
///
/// Requires macOS 15 / iOS 18 (MLState + multifunction models). Shipped
/// voices are the 27 upstream presets (`alba` default).
/// Callers can also speak a voice of their own by passing its conditioning
/// rows (`PocketTtsSmallConditioning`) to the `conditioning:` overloads.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
public actor PocketTtsSmallManager {

    private static let logger = AppLogger(category: "PocketTtsSmallManager")

    public nonisolated let model: PocketTtsSmallModel
    public nonisolated let computeUnits: MLComputeUnits
    public nonisolated let mimiPrecision: PocketTtsSmallMimiPrecision
    public private(set) var defaultVoice: String

    private var constants: PocketTtsSmallConstants?
    private var models: PocketTtsSmallModels?
    private var packDirectory: URL?
    private var voiceCache: [String: PocketTtsSmallEngine.Fp16Snapshot] = [:]

    public init(
        model: PocketTtsSmallModel = .english7m,
        defaultVoice: String? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        mimiPrecision: PocketTtsSmallMimiPrecision = .fp32
    ) {
        self.model = model
        self.defaultVoice = defaultVoice ?? model.defaultVoice
        self.computeUnits = computeUnits
        self.mimiPrecision = mimiPrecision
    }

    public var isInitialized: Bool { models != nil }
    public var arch: PocketTtsSmallArch? { constants?.arch }

    /// Download (if needed) and load the pack.
    public func initialize(directory: URL? = nil, progressHandler: ProgressHandler? = nil) async throws {
        let pack = try await PocketTtsSmallDownloader.ensurePack(
            model, directory: directory, progressHandler: progressHandler)
        try load(from: pack)
    }

    /// Load from a local pack directory (`build/<model>/` of the converter).
    public func load(from packDirectory: URL) throws {
        let constants = try PocketTtsSmallConstants.load(from: packDirectory)
        let t0 = Date()
        self.models = try PocketTtsSmallModels(
            packDirectory: packDirectory, computeUnits: computeUnits, mimiPrecision: mimiPrecision)
        self.constants = constants
        self.packDirectory = packDirectory
        self.voiceCache = [:]
        Self.logger.info(
            "Loaded PocketTTS \(model.rawValue) (\(constants.arch.dModel)-d, \(constants.arch.numLayers)L, "
                + "mimi \(mimiPrecision.rawValue)) in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
    }

    public func setDefaultVoice(_ voice: String) { defaultVoice = voice }

    /// Shipped voices available in the loaded pack.
    public func availableVoices() throws -> [String] {
        guard let constants else { throw PocketTTSError.modelNotFound("PocketTTS small model not initialized") }
        return constants.voiceNames()
    }

    /// Open a streaming session with a shipped voice.
    public func makeSession(
        voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> PocketTtsSmallSession {
        // Split into snapshot + openSession so the conditioning entry point shares the session start
        // wangqi modified 2026-11-07
        let (engine, constants) = try makeEngine()
        let snapshot = try await presetSnapshot(voice ?? defaultVoice, engine: engine, constants: constants)
        return await openSession(
            engine: engine, constants: constants, snapshot: snapshot, temperature: temperature, seed: seed)
    }

    // Caller-supplied conditioning entry points (the app's imported voice files) // wangqi modified 2026-11-07
    /// Open a streaming session that speaks caller-supplied voice conditioning.
    /// The rows are prefilled into an empty cache once per `cacheKey`; pass a
    /// new key whenever the rows change.
    public func makeSession(
        conditioning: PocketTtsSmallConditioning, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> PocketTtsSmallSession {
        let (engine, constants) = try makeEngine()
        let snapshot = try await conditioningSnapshot(conditioning, engine: engine, constants: constants)
        return await openSession(
            engine: engine, constants: constants, snapshot: snapshot, temperature: temperature, seed: seed)
    }

    /// Synthesize caller-supplied voice conditioning to 80 ms frames.
    public func synthesizeStreaming(
        text: String, conditioning: PocketTtsSmallConditioning, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error> {
        let session = try await makeSession(conditioning: conditioning, temperature: temperature, seed: seed)
        session.enqueue(text)
        session.finish()
        return session.frames
    }

    /// Synthesize caller-supplied voice conditioning to 24 kHz 16-bit WAV data.
    public func synthesize(
        text: String, conditioning: PocketTtsSmallConditioning, temperature: Float? = nil, seed: UInt64? = nil,
        deEss: Bool = true
    ) async throws -> Data {
        let t0 = Date()
        let frames = try await synthesizeStreaming(
            text: text, conditioning: conditioning, temperature: temperature, seed: seed)
        let samples = try await collectSamples(frames, startedAt: t0, deEss: deEss)
        return try AudioWAV.data(from: samples, sampleRate: Double(PocketTtsConstants.audioSampleRate))
    }

    // Shared session start for the voice and conditioning entry points // wangqi modified 2026-11-07
    /// One engine (one MLState) per session: concurrent sessions must not
    /// share the mutable KV / Mimi state. The MLModel handles are shared.
    private func makeEngine() throws -> (PocketTtsSmallEngine, PocketTtsSmallConstants) {
        guard let models, let constants else {
            throw PocketTTSError.modelNotFound("PocketTTS small model not initialized")
        }
        return (try PocketTtsSmallEngine(models: models, constants: constants), constants)
    }

    /// The KV snapshot of a shipped voice, cached by name.
    private func presetSnapshot(
        _ name: String, engine: PocketTtsSmallEngine, constants: PocketTtsSmallConstants
    ) async throws -> PocketTtsSmallEngine.Fp16Snapshot {
        if let cached = voiceCache[name] { return cached }
        let snapshot: PocketTtsSmallEngine.Fp16Snapshot
        if constants.hasKVSnapshot(name) {
            snapshot = try await engine.fp16Snapshot(from: constants.loadVoice(name))
        } else {
            // A voice shipped as conditioning rows: prefill it into an empty cache
            // once and keep the resulting KV state as this voice's snapshot.
            let voice = try constants.loadVoiceConditioning(name)
            snapshot = try await prefillSnapshot(rows: voice.rows, frameCount: voice.count, engine: engine)
        }
        voiceCache[name] = snapshot
        return snapshot
    }

    /// The KV snapshot of caller-supplied conditioning, cached under a
    /// `conditioning:` prefix so it never collides with a shipped voice name.
    private func conditioningSnapshot(
        _ conditioning: PocketTtsSmallConditioning, engine: PocketTtsSmallEngine,
        constants: PocketTtsSmallConstants
    ) async throws -> PocketTtsSmallEngine.Fp16Snapshot {
        let dModel = constants.arch.dModel
        guard conditioning.frameCount > 0, conditioning.rows.count == conditioning.frameCount * dModel else {
            Self.logger.warning(
                "Rejected conditioning: \(conditioning.rows.count) floats, expected "
                    + "\(conditioning.frameCount)x\(dModel)")
            throw PocketTtsConstantsLoader.LoadError.malformed(
                "conditioning: \(conditioning.rows.count) floats, expected \(conditioning.frameCount)x\(dModel)")
        }
        let key = "conditioning:" + conditioning.cacheKey
        if let cached = voiceCache[key] { return cached }
        let snapshot = try await prefillSnapshot(
            rows: conditioning.rows, frameCount: conditioning.frameCount, engine: engine)
        voiceCache[key] = snapshot
        return snapshot
    }

    private func prefillSnapshot(
        rows: [Float], frameCount: Int, engine: PocketTtsSmallEngine
    ) async throws -> PocketTtsSmallEngine.Fp16Snapshot {
        try await engine.resetToZero()
        try await engine.prefill(flatConditioning: rows, tokenCount: frameCount)
        return try await engine.captureSnapshot()
    }

    private func openSession(
        engine: PocketTtsSmallEngine, constants: PocketTtsSmallConstants,
        snapshot: PocketTtsSmallEngine.Fp16Snapshot, temperature: Float?, seed: UInt64?
    ) async -> PocketTtsSmallSession {
        var textNormalizer: (@Sendable (String) -> String)?
        if model == .phonon31m {
            textNormalizer = { @Sendable text in PhononTextNormalizer.normalize(text) }
        }
        let session = PocketTtsSmallSession(
            engine: engine, constants: constants, voiceSnapshot: snapshot,
            temperature: temperature, seed: seed ?? UInt64.random(in: 0...UInt64.max),
            textNormalizer: textNormalizer)
        await session.start()
        return session
    }

    /// Synthesize to 80 ms frames.
    public func synthesizeStreaming(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws -> AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error> {
        let session = try await makeSession(voice: voice, temperature: temperature, seed: seed)
        session.enqueue(text)
        session.finish()
        return session.frames
    }

    /// Synthesize to 24 kHz 16-bit WAV data.
    public func synthesize(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil, deEss: Bool = true
    ) async throws -> Data {
        let samples = try await synthesizeSamples(
            text: text, voice: voice, temperature: temperature, seed: seed, deEss: deEss)
        return try AudioWAV.data(from: samples, sampleRate: Double(PocketTtsConstants.audioSampleRate))
    }

    /// Synthesize to raw Float32 samples at 24 kHz.
    public func synthesizeSamples(
        text: String, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil, deEss: Bool = true
    ) async throws -> [Float] {
        // One frame loop shared with the conditioning path (de-ess + RTFx log) // wangqi modified 2026-11-07
        let t0 = Date()
        let frames = try await synthesizeStreaming(text: text, voice: voice, temperature: temperature, seed: seed)
        return try await collectSamples(frames, startedAt: t0, deEss: deEss)
    }

    private func collectSamples(
        _ stream: AsyncThrowingStream<PocketTtsSynthesizer.AudioFrame, Error>, startedAt t0: Date, deEss: Bool
    ) async throws -> [Float] {
        var samples: [Float] = []
        var frames = 0
        for try await frame in stream {
            samples.append(contentsOf: frame.samples)
            frames += 1
        }
        if deEss {
            AudioPostProcessor.applyTtsPostProcessing(
                &samples, sampleRate: Float(PocketTtsConstants.audioSampleRate), deEssAmount: -3.0, smoothing: false)
        }
        let secs = Double(samples.count) / Double(PocketTtsConstants.audioSampleRate)
        let elapsed = Date().timeIntervalSince(t0)
        Self.logger.info(
            "\(model.rawValue): \(frames) frames, \(String(format: "%.2f", secs))s audio in "
                + "\(String(format: "%.0f", elapsed * 1000))ms (RTFx \(String(format: "%.1f", secs / max(elapsed, 1e-6))))"
        )
        return samples
    }

    public func synthesizeToFile(
        text: String, outputURL: URL, voice: String? = nil, temperature: Float? = nil, seed: UInt64? = nil
    ) async throws {
        let wav = try await synthesize(text: text, voice: voice, temperature: temperature, seed: seed)
        try wav.write(to: outputURL)
    }
}

// Caller-supplied voice conditioning for imported voices (the app's Voice Clone voice files)
// wangqi modified 2026-11-07
/// Voice conditioning supplied by the caller: `emb` rows `[frameCount, dModel]`
/// flattened row-major. `cacheKey` names these rows in the manager's voice
/// cache and must change whenever the rows do.
@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)
public struct PocketTtsSmallConditioning: Sendable {
    public let rows: [Float]
    public let frameCount: Int
    public let cacheKey: String

    public init(rows: [Float], frameCount: Int, cacheKey: String) {
        self.rows = rows
        self.frameCount = frameCount
        self.cacheKey = cacheKey
    }
}

/// Downloads a small-model pack (`v4/<model>/`) from HuggingFace: the MLState
/// package, the fp32 `mimi_decoder.mlmodelc`, the constants and the voices —
/// the other fp32 I/O models and the `.mlpackage` sources are skipped.
public enum PocketTtsSmallDownloader {
    private static let logger = AppLogger(category: "PocketTtsSmallDownloader")

    /// Files a small-model pack needs on disk. The compiled bundle and the
    /// constants directory are checked through files INSIDE them, because
    /// `ModelHub.download` creates the directories as soon as their first file
    /// lands and an interrupted download must not look complete.
    static let requiredFiles: [String] = [
        ModelNames.PocketTTS.pocketStateFile + "/coremldata.bin",
        ModelNames.PocketTTS.pocketStateFile + "/model.mil",
        ModelNames.PocketTTS.mimiDecoderFile + "/coremldata.bin",
        ModelNames.PocketTTS.mimiDecoderFile + "/model.mil",
        "arch.json", "pocket_state.json",
        ModelNames.PocketTTS.constantsBinDir + "/tokenizer.model",
        ModelNames.PocketTTS.constantsBinDir + "/text_embed_table.bin",
    ]

    /// `requiredFiles` plus the model's default voice (a KV snapshot in `constants_bin/`).
    static func requiredFiles(for model: PocketTtsSmallModel) -> [String] {
        requiredFiles + [ModelNames.PocketTTS.constantsBinDir + "/\(model.defaultVoice).safetensors"]
    }

    public static func ensurePack(
        _ model: PocketTtsSmallModel, directory: URL? = nil, progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let base = try directory ?? TtsCacheDirectory.ensure()
        let repoDir = base.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(Repo.pocketTts.folderName)
        let pack = repoDir.appendingPathComponent(model.repoSubdirectory)
        if requiredFiles(for: model).allSatisfy({
            FileManager.default.fileExists(atPath: pack.appendingPathComponent($0).path)
        }) {
            logger.info("PocketTTS \(model.rawValue) pack found in cache")
            return pack
        }
        logger.info("Downloading PocketTTS \(model.rawValue) pack (\(model.repoSubdirectory))...")
        try await ModelHub.download(
            .pocketTts, subdirectory: model.repoSubdirectory, to: repoDir,
            progressHandler: progressHandler, shouldSkip: { shouldSkipAsset(at: $0) })
        return pack
    }

    /// Keep `pocket_state.mlmodelc`, `mimi_decoder.mlmodelc`, the JSON
    /// descriptors and `constants_bin/`; skip the other fp32 I/O `.mlmodelc`
    /// bundles, every `.mlpackage` and the `constants/` intermediates.
    static func shouldSkipAsset(at path: String) -> Bool {
        let components = path.split(separator: "/").map(String.init)
        let basename = components.last ?? path
        if basename == ".DS_Store" || basename.hasSuffix(".wav") { return true }
        if components.contains(where: { $0.hasSuffix(".mlpackage") }) { return true }
        if let bundle = components.first(where: { $0.hasSuffix(".mlmodelc") }),
            bundle != ModelNames.PocketTTS.pocketStateFile, bundle != ModelNames.PocketTTS.mimiDecoderFile
        {
            return true
        }
        if components.contains("constants") { return true }
        if basename == "manifest.json" { return true }
        return false
    }
}
#endif
