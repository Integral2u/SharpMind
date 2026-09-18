namespace SharpMind.Inference.Grammar;

/// <summary>
/// A parsed and compiled GBNF grammar. Parsing is expensive relative to a
/// generation step, so parse once and call
/// <see cref="CreateConstraint"/> for each generation that must obey it.
/// </summary>
public sealed class GbnfGrammar
{
    private readonly GrammarNfa _nfa;

    private GbnfGrammar(GrammarNfa nfa) => _nfa = nfa;

    /// <summary>Parses and compiles a llama.cpp-style GBNF grammar.</summary>
    /// <exception cref="GrammarException">The grammar is malformed.</exception>
    public static GbnfGrammar Parse(string gbnf)
        => new(new GbnfCompiler(GbnfParser.Parse(gbnf)).Compile());

    internal GrammarNfa Nfa => _nfa;

    /// <summary>
    /// Creates a constraint for one generation. <paramref name="stopTokenIds"/>
    /// are the tokens that end the turn (usually the tokenizer's EOS/EOG ids);
    /// they are offered only once the grammar is complete, so the model can
    /// never stop mid-structure.
    /// </summary>
    public IGrammarConstraint CreateConstraint(TokenByteTable tokenBytes, IEnumerable<int>? stopTokenIds = null)
        => new GrammarConstraint(this, tokenBytes, stopTokenIds);
}

/// <summary>Thrown when a grammar cannot be parsed or compiled.</summary>
public sealed class GrammarException : Exception
{
    public GrammarException(string message) : base(message) { }

    public GrammarException(string message, Exception innerException)
        : base(message, innerException) { }
}