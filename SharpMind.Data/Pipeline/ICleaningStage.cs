namespace SharpMind.Data.Pipeline;

/// <summary>
/// A single processing stage in the cleaning DAG.
///
/// Returns the transformed document, or null to discard it entirely.
/// Stages are pure functions — they must not maintain mutable state that
/// would cause non-deterministic output across threads or runs.
/// </summary>
public interface ICleaningStage
{
    /// <summary>
    /// Transforms <paramref name="document"/>.
    /// Return null to drop the document from the stream entirely.
    /// </summary>
    string? Process(string document);

    /// <summary>Short name used in pipeline descriptions and diagnostics.</summary>
    string Name { get; }
}
