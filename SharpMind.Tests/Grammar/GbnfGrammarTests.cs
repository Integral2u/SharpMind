using SharpMind.Inference.Grammar;
using Xunit;

namespace SharpMind.Tests.Grammar;

/// <summary>
/// Milestone-2 tests for parsing and compiling GBNF grammars.
/// </summary>
public sealed class GbnfGrammarTests
{
    private const string AllConstructs = """
        # a grammar exercising every parser feature
        root      ::= "'" name "'" "=" ws (digits | word) trailing?
        name      ::= [a-zA-Z_] [a-zA-Z0-9_]*
        digits    ::= [0-9]+
        word      ::= "\"" [^"]* "\""
        trailing  ::= " " hex+
        hex       ::= [0-9a-fA-F]{2}
        ws        ::= [ \t]*
        """;

    [Fact]
    public void Parse_AllConstructs_Compiles()
    {
        var grammar = GbnfGrammar.Parse(AllConstructs);
        Assert.NotNull(grammar);

        var table = TestTokenizer.ByteTable;
        var constraint = grammar.CreateConstraint(table, [TestTokenizer.EosId]);
        Assert.NotNull(constraint);
        Assert.False(constraint.IsDead);
    }

    [Fact]
    public void Parse_EmptyGrammar_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("   # nothing here\n"));

    [Fact]
    public void Parse_MissingRoot_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("start ::= \"a\""));

    [Fact]
    public void Parse_UndefinedRule_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= missing"));

    [Fact]
    public void Parse_DuplicateRule_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= \"a\"\nroot ::= \"b\""));

    [Fact]
    public void Parse_UnterminatedLiteral_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= \"abc"));

    [Fact]
    public void Parse_UnclosedGroup_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= (\"a\""));

    [Fact]
    public void Parse_ReversedRepetitionRange_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= \"a\"{3,1}"));

    [Fact]
    public void Parse_InvalidEscape_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= \"\\q\""));

    [Fact]
    public void Parse_UnknownQuantifierTarget_Throws()
        => Assert.Throws<GrammarException>(() => GbnfGrammar.Parse("root ::= *"));
}