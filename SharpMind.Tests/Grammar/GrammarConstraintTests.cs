using System.Text;
using SharpMind.Inference.Grammar;
using Xunit;

namespace SharpMind.Tests.Grammar;

/// <summary>
/// Milestone-2 tests for <see cref="GrammarConstraint"/>: using the byte-level
/// test vocabulary, generation is driven one byte at a time so the mask can be
/// checked exactly, including EOS gating and dead-end detection.
/// </summary>
public sealed class GrammarConstraintTests
{
    private static IGrammarConstraint Constrain(string gbnf, params int[] stops)
        => GbnfGrammar.Parse(gbnf).CreateConstraint(
            TestTokenizer.ByteTable,
            stops.Length == 0 ? [TestTokenizer.EosId] : stops);

    private static int Id(byte b) => TestTokenizer.IdFor(b);
    private static int Id(char ascii) => TestTokenizer.IdFor((byte)ascii);

    private static float[] Fresh() => new float[TestTokenizer.Count];

    /// <summary>Feeds a byte string, asserting every byte stays inside the grammar.</summary>
    private static void AcceptText(IGrammarConstraint constraint, string text)
    {
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            constraint.Apply(Fresh());
            int id = Id(b);
            Assert.True(constraint.IsAllowed(id), $"byte 0x{b:X2} was masked while emitting '{text}'");
            constraint.Accept(id);
        }
    }

    // ─── Literal ───────────────────────────────────────────────────────────

    [Fact]
    public void Literal_AllowsOnlyTheNextByte()
    {
        var constraint = Constrain("root ::= \"abc\"");
        float[] logits = Fresh();
        constraint.Apply(logits);

        Assert.False(constraint.CanStop);
        Assert.Equal(1, ((GrammarConstraint)constraint).AllowedTokenCount);
        Assert.True(constraint.IsAllowed(Id('a')));
        Assert.False(constraint.IsAllowed(Id('b')));
        Assert.False(constraint.IsAllowed(TestTokenizer.EosId));

        Assert.Equal(0f, logits[Id('a')]);
        Assert.True(IsMasked(logits[Id('b')]));
        Assert.True(IsMasked(logits[TestTokenizer.EosId]));
    }

    [Fact]
    public void Literal_CompletesThenGatesEosUntilDone()
    {
        var constraint = Constrain("root ::= \"ab\"");
        constraint.Apply(Fresh());
        Assert.False(constraint.CanStop);
        Assert.False(constraint.IsAllowed(TestTokenizer.EosId));

        constraint.Accept(Id('a'));
        constraint.Apply(Fresh());
        Assert.False(constraint.CanStop);
        Assert.False(constraint.IsAllowed(TestTokenizer.EosId));
        Assert.True(constraint.IsAllowed(Id('b')));

        constraint.Accept(Id('b'));
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.True(constraint.IsAllowed(TestTokenizer.EosId));
        Assert.False(constraint.IsAllowed(Id('a')));
    }

    // ─── Classes ───────────────────────────────────────────────────────────

    [Fact]
    public void Class_Range_AllowsMembersOnly()
    {
        var constraint = Constrain("root ::= [a-c]+");
        constraint.Apply(Fresh());

        Assert.True(constraint.IsAllowed(Id('a')));
        Assert.True(constraint.IsAllowed(Id('b')));
        Assert.True(constraint.IsAllowed(Id('c')));
        Assert.False(constraint.IsAllowed(Id('d')));
        Assert.False(constraint.CanStop); // '+' requires at least one member
    }

    [Fact]
    public void Class_Negated_ExcludesMembers()
    {
        var constraint = Constrain("root ::= [^0-9]+");
        constraint.Apply(Fresh());

        Assert.True(constraint.IsAllowed(Id('a')));
        Assert.True(constraint.IsAllowed(Id(' ')));
        Assert.False(constraint.IsAllowed(Id('5')));
        Assert.False(constraint.CanStop);
    }

    [Fact]
    public void Class_MultiByte_MatchesUtf8Sequence()
    {
        // é = U+00E9 -> UTF-8 0xC3 0xA9
        var constraint = Constrain("root ::= [\u00E9]");
        constraint.Apply(Fresh());
        Assert.False(constraint.IsAllowed(Id(0xA9)));
        Assert.True(constraint.IsAllowed(Id(0xC3)));

        constraint.Accept(Id(0xC3));
        constraint.Apply(Fresh());
        Assert.True(constraint.IsAllowed(Id(0xA9)));

        constraint.Accept(Id(0xA9));
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
    }

    // ─── Repetition / alternation ──────────────────────────────────────────

    [Fact]
    public void BoundedRepeat_EnforcesMinAndMax()
    {
        var constraint = Constrain("root ::= \"a\"{2,4}");

        constraint.Apply(Fresh());
        Assert.False(constraint.CanStop);
        Assert.True(constraint.IsAllowed(Id('a')));
        Assert.False(constraint.IsAllowed(TestTokenizer.EosId));

        AcceptText(constraint, "a");
        constraint.Apply(Fresh());
        Assert.False(constraint.CanStop);

        AcceptText(constraint, "a");
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.True(constraint.IsAllowed(TestTokenizer.EosId));
        Assert.True(constraint.IsAllowed(Id('a')));

        AcceptText(constraint, "aa");
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.False(constraint.IsAllowed(Id('a')));
    }

    [Fact]
    public void Alternation_Star_AllowsBothAndCanStopImmediately()
    {
        var constraint = Constrain("root ::= (\"a\" | \"b\")*");
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.True(constraint.IsAllowed(Id('a')));
        Assert.True(constraint.IsAllowed(Id('b')));
        Assert.False(constraint.IsAllowed(Id('c')));

        AcceptText(constraint, "abba");
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
    }

    // ─── Recursion ─────────────────────────────────────────────────────────

    [Fact]
    public void RecursiveArray_EmitsNestedAndStops()
    {
        var constraint = Constrain("""
            root  ::= array
            array ::= "[" (array ("," array)*)? "]"
            """);

        constraint.Apply(Fresh());
        Assert.True(constraint.IsAllowed(Id('[')));
        Assert.False(constraint.IsAllowed(Id(']')));
        Assert.False(constraint.IsAllowed(Id(',')));
        Assert.False(constraint.IsAllowed(Id('x')));

        AcceptText(constraint, "[");
        constraint.Apply(Fresh());
        Assert.True(constraint.IsAllowed(Id('['))); // nested array
        Assert.True(constraint.IsAllowed(Id(']')));  // or close immediately

        AcceptText(constraint, "[],[[]]]");
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.True(constraint.IsAllowed(TestTokenizer.EosId));
        Assert.False(constraint.IsDead);
    }

    // ─── JSON integration (recursion + classes + repetition) ───────────────

    private const string JsonGrammar = """
        root    ::= value
        value   ::= object | array | string | number | "true" | "false" | "null"
        object  ::= "{" ws (member (ws "," ws member)*)? ws "}"
        member  ::= string ws ":" ws value
        array   ::= "[" ws (value (ws "," ws value)*)? ws "]"
        string  ::= "\"" chars "\""
        chars   ::= [^"\\]*
        number  ::= "-"? [0-9]+
        ws      ::= [ \t\n\r]*
        """;

    private const string JsonTarget = "{\"a\":1,\"b\":[2,3],\"c\":true}";

    [Fact]
    public void JsonObject_EmitsValidJsonAndStops()
    {
        var constraint = Constrain(JsonGrammar);
        AcceptText(constraint, JsonTarget);
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.False(constraint.IsDead);
        Assert.True(constraint.IsAllowed(TestTokenizer.EosId));
    }

    [Fact]
    public void JsonObject_ConstrainedGreedy_PicksExpectedBytes()
    {
        var constraint = Constrain(JsonGrammar);
        var produced = new StringBuilder();
        foreach (byte b in Encoding.UTF8.GetBytes(JsonTarget))
        {
            float[] logits = Fresh();
            logits[Id(b)] = 10f; // desired byte wins unless the grammar forbids it
            constraint.Apply(logits);
            int best = ArgMax(logits);
            Assert.Equal(Id(b), best);
            constraint.Accept(best);
            produced.Append((char)b);
        }

        Assert.Equal(JsonTarget, produced.ToString());
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
    }

    // ─── Dead ends / violations ────────────────────────────────────────────

    [Fact]
    public void ViolatingToken_MarksDeadAndStaysDead()
    {
        var constraint = Constrain("root ::= \"ab\"");
        constraint.Apply(Fresh());
        constraint.Accept(Id('a'));

        constraint.Accept(Id('x')); // not part of the grammar
        Assert.True(constraint.IsDead);

        float[] logits = Fresh();
        constraint.Apply(logits);
        Assert.True(constraint.IsDead);
        Assert.All(logits, value => Assert.True(IsMasked(value)));
    }

    [Fact]
    public void InvalidStopId_IsIgnored()
    {
        var constraint = Constrain("root ::= \"a\"", -1, 999_999, TestTokenizer.EosId);
        constraint.Apply(Fresh());
        Assert.True(constraint.IsAllowed(Id('a')));

        constraint.Accept(Id('a'));
        constraint.Apply(Fresh());
        Assert.True(constraint.CanStop);
        Assert.True(constraint.IsAllowed(TestTokenizer.EosId));
    }

    // ─── Helpers ───────────────────────────────────────────────────────────

    private static bool IsMasked(float value) => value <= GrammarConstraint.MaskedLogit;

    private static int ArgMax(float[] logits)
    {
        int best = -1;
        float bestValue = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (IsMasked(logits[i]))
                continue;
            if (logits[i] > bestValue)
            {
                bestValue = logits[i];
                best = i;
            }
        }
        Assert.True(best >= 0, "no token was allowed");
        return best;
    }
}