using System.Runtime.InteropServices;
using SharpMind.Inference;

namespace SharpMind.Tests.Inference;

public class RepetitionPenaltyTests
{
    [Fact]
    public void PenalizesOncePerDistinctToken_NotPerOccurrence()
    {
        // A common token appearing 5x must be scaled ONCE, not penalty^5. If it were
        // scaled per occurrence, "the"(id 10) would vanish from the distribution.
        var logits = new float[50];
        for (int i = 0; i < logits.Length; i++) logits[i] = 10f;
        ReadOnlySpan<int> ids = [10, 20, 30, 40, 10, 10, 10, 10, 20];
        float penalty = 1.25f;

        RepetitionPenalty.Apply(logits, ids, penalty, new HashSet<int>());

        Assert.Equal(10f / penalty, logits[10], 4);   // once
        Assert.Equal(10f / penalty, logits[20], 4);   // once
        Assert.Equal(10f / penalty, logits[30], 4);   // once
        Assert.Equal(10f / penalty, logits[40], 4);   // once
        Assert.Equal(10f, logits[9], 4);              // untouched
    }

    [Fact]
    public void NegativeLogitsAreBoostedInMagnitude()
    {
        var logits = new float[8];
        logits[1] = -5f;   // repeated negative token
        logits[2] = -5f;   // distinct negative token
        ReadOnlySpan<int> ids = [1, 1, 1, 2];

        RepetitionPenalty.Apply(logits, ids, 1.1f, new HashSet<int>());

        Assert.Equal(-5f * 1.1f, logits[1], 4);   // once (distinct)
        Assert.Equal(-5f * 1.1f, logits[2], 4);   // once (distinct)
    }

    [Fact]
    public void SharedSeenSetSpansMultipleCalls()
    {
        // A token present in both prompt and generated must be penalized once overall.
        var logits = new float[] { 0, 8f, 8f, 8f };
        var seen = new HashSet<int>();

        RepetitionPenalty.Apply(logits, [1, 2, 1], 2f, seen);
        RepetitionPenalty.Apply(logits, [2, 3], 2f, seen);

        Assert.Equal(4f, logits[1], 4);   // 8/2 once
        Assert.Equal(4f, logits[2], 4);   // 8/2 once despite appearing in both calls
        Assert.Equal(4f, logits[3], 4);   // 8/2 once
    }
}