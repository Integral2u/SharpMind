namespace SharpMind.Inference.Chat;

public interface IContextCompactor
{
    Task<bool> ShouldCompactAsync(CompactionContext context, CancellationToken ct);
    Task<bool> CompactAsync(CompactionContext context, CancellationToken ct);
}
