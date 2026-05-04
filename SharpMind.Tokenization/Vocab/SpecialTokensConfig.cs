namespace SharpMind.Tokenization.Vocab;

/// <summary>
/// Configuration passed to <see cref="SpecialTokens"/> constructor.
/// All fields are optional — unset fields use the defaults.
/// </summary>
public sealed record SpecialTokensConfig
{
    public string? Unk { get; init; }
    public string? Bos { get; init; }
    public string? Eos { get; init; }
    public string? Pad { get; init; }
    public IReadOnlyList<string> Additional { get; init; } = [];
}
