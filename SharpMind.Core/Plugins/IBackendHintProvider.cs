namespace SharpMind.Core.Plugins;

/// <summary>
/// Optional plugin capability describing which backend the plugin would actually run on this
/// machine, purely for display (e.g. <c>"OpenCL"</c>, <c>"CUDA · cuBLAS"</c>, <c>"CUDA"</c>, or
/// <c>null</c> when no real accelerator is available). The CUI appends this to the plugin's name in
/// accelerator selectors so the user sees the real backend instead of a bare plugin name — without
/// the CUI referencing the plugin's assembly (issue #13).
/// </summary>
public interface IBackendHintProvider
{
    /// <summary>Human-readable backend name, or <c>null</c> when no real accelerator could run.</summary>
    string? BackendHint();
}
