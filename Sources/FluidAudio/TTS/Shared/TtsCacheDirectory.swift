import Foundation

/// Cross-backend TTS cache root.
///
/// Mirrors the cache root previously exposed by the now-removed
/// `TtsModels.cacheDirectoryURL()` so callers (currently `G2PModel` and
/// `MultilingualG2PModel`) can still resolve
/// `~/.cache/fluidaudio/Models/<repo>/` paths without depending on a
/// backend-specific manager.
///
/// macOS resolves to `~/.cache/fluidaudio`; other platforms use the
/// application-support directory under `fluidaudio/`.
///
/// On iOS/iPadOS this deliberately avoids `.cachesDirectory`: `Library/Caches`
/// is reclaimable by the system under disk pressure, so models (hundreds of MB)
/// could be silently purged while backgrounded and require a full re-download.
/// `.applicationSupportDirectory` is the recommended location for re-obtainable
/// but expensive-to-replace data, matching the ASR side.
public enum TtsCacheDirectory {

    /// Redirects every TTS asset lookup to a caller-supplied directory (the host app's own
    /// download folder) instead of the default `fluidaudio` cache. Set once at startup, before
    /// any TTS loader runs. Replaces the pre-0.16 `TtsModels.overrideCacheDirectory`, which
    /// died with the CoreML Kokoro backend.
    /// wangqi modified 2026-09-18
    public static var overrideDirectory: URL?

    /// Ensures the platform-appropriate cache root exists and returns it.
    public static func ensure() throws -> URL {
        // App-supplied override wins; falls back to the default fluidaudio cache.
        // The host app creates and owns this directory, so it is returned as-is.
        // wangqi modified 2026-09-18
        if let override = overrideDirectory {
            return override
        }

        let base: URL
        #if os(macOS)
        base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            throw TTSError.processingFailed("Failed to locate application support directory")
        }
        base = first
        #endif

        let cacheDirectory = base.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory
    }
}
