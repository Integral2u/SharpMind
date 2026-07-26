namespace SharpMind.Inference.Chat;

public interface IContextCompactor
{
    string Name { get; }
    Task<bool> ShouldCompactAsync(CompactionContext context, CancellationToken ct);
    Task<bool> CompactAsync(CompactionContext context, CancellationToken ct);
}
