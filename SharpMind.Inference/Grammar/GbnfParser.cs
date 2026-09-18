using System.Text;

namespace SharpMind.Inference.Grammar;

// ─── AST ───────────────────────────────────────────────────────────────────

internal abstract class GNode;

/// <summary>A literal string, stored as UTF-8 bytes (matching is byte-level).</summary>
internal sealed class GLiteral : GNode
{
    public required byte[] Bytes { get; init; }
}

/// <summary>
/// A character class <c>[..]</c> or negated <c>[^..]</c>. Single-byte code
/// points live in <see cref="Allowed"/>; multi-byte code points in a positive
/// class become alternative byte sequences in <see cref="Sequences"/>. For a
/// negated class, <see cref="Allowed"/> holds the excluded single bytes and
/// multi-byte exclusions are ignored (see the parser for the rationale).
/// </summary>
internal sealed class GClass : GNode
{
    public bool Negated { get; init; }
    public bool[] Allowed { get; } = new bool[256];
    public List<byte[]> Sequences { get; } = [];
}

internal sealed class GSequence : GNode
{
    public GNode[] Items { get; init; } = [];
}

internal sealed class GAlternation : GNode
{
    public GNode[] Options { get; init; } = [];
}

internal sealed class GReference : GNode
{
    public required string Name { get; init; }
}

/// <summary>Repetition of an item. <see cref="Max"/> of -1 means unbounded.</summary>
internal sealed class GRepeat : GNode
{
    public required GNode Item { get; init; }
    public int Min { get; init; }
    public int Max { get; init; }
}

// ─── Parser ────────────────────────────────────────────────────────────────

/// <summary>
/// Recursive-descent parser for llama.cpp-style GBNF grammars:
/// <code>
/// root  ::= "{" ws "\"x\"" ws ":" ws [0-9]+ ws "}"
/// ws    ::= [ \t\n]*
/// </code>
/// Supports literals with escapes, character classes (including ranges and
/// negation), groups, rule references, alternation, and the <c>* + ?</c> and
/// <c>{m}</c> <c>{m,}</c> <c>{m,n}</c> repetition suffixes. <c>#</c> starts a
/// comment to end of line.
/// </summary>
internal sealed class GbnfParser
{
    private const int MaxPositiveClassSequences = 4096;

    private readonly string _s;
    private int _i;

    private GbnfParser(string grammar) => _s = grammar;

    public static Dictionary<string, GNode> Parse(string grammar)
    {
        ArgumentNullException.ThrowIfNull(grammar);
        var parser = new GbnfParser(grammar);
        return parser.ParseRules();
    }

    private Dictionary<string, GNode> ParseRules()
    {
        var rules = new Dictionary<string, GNode>(StringComparer.Ordinal);
        SkipTrivia();
        while (!Eof)
        {
            string name = ParseName();
            SkipTrivia();
            Expect("::=");
            SkipTrivia();
            GNode body = ParseAlternation();
            if (!rules.TryAdd(name, body))
                throw Error($"duplicate rule '{name}'");
            SkipTrivia();
        }
        if (rules.Count == 0)
            throw Error("grammar defines no rules");
        return rules;
    }

    private GNode ParseAlternation()
    {
        var options = new List<GNode> { ParseSequence() };
        SkipTrivia();
        while (!Eof && Peek() == '|')
        {
            _i++;
            SkipTrivia();
            options.Add(ParseSequence());
            SkipTrivia();
        }
        return options.Count == 1 ? options[0] : new GAlternation { Options = [.. options] };
    }

    private GNode ParseSequence()
    {
        var items = new List<GNode>();
        while (true)
        {
            SkipTrivia();
            if (Eof || Peek() == '|' || Peek() == ')' || AtRuleStart())
                break;
            items.Add(ParseItem());
        }
        return items.Count switch
        {
            0 => new GSequence(),
            1 => items[0],
            _ => new GSequence { Items = [.. items] },
        };
    }

    private GNode ParseItem()
    {
        GNode atom = ParseAtom();
        SkipTrivia();
        if (Eof) return atom;

        switch (Peek())
        {
            case '*':
                _i++;
                return new GRepeat { Item = atom, Min = 0, Max = -1 };
            case '+':
                _i++;
                return new GRepeat { Item = atom, Min = 1, Max = -1 };
            case '?':
                _i++;
                return new GRepeat { Item = atom, Min = 0, Max = 1 };
            case '{':
                return ParseBoundedRepeat(atom);
            default:
                return atom;
        }
    }

    private GNode ParseBoundedRepeat(GNode atom)
    {
        _i++; // '{'
        SkipTrivia();
        int min = ParseNumber();
        SkipTrivia();
        int max = min;
        if (!Eof && Peek() == ',')
        {
            _i++;
            SkipTrivia();
            max = (!Eof && char.IsDigit(Peek())) ? ParseNumber() : -1;
        }
        SkipTrivia();
        Expect('}');
        if (max != -1 && max < min)
            throw Error($"invalid repetition range {{{min},{max}}}");
        return new GRepeat { Item = atom, Min = min, Max = max };
    }

    private GNode ParseAtom()
    {
        if (Eof) throw Error("unexpected end of grammar");
        return Peek() switch
        {
            '"' => ParseLiteral(),
            '[' => ParseClass(),
            '(' => ParseGroup(),
            _ when IsNameStart(Peek()) => new GReference { Name = ParseName() },
            _ => throw Error($"unexpected character '{Peek()}'"),
        };
    }

    private GNode ParseGroup()
    {
        _i++; // '('
        SkipTrivia();
        GNode body = ParseAlternation();
        SkipTrivia();
        Expect(')');
        return body;
    }

    private GNode ParseLiteral()
    {
        _i++; // '"'
        var bytes = new List<byte>();
        while (!Eof && Peek() != '"')
        {
            int cp = ParseEscapedCodePoint();
            bytes.AddRange(Encoding.UTF8.GetBytes(char.ConvertFromUtf32(cp)));
        }
        Expect('"');
        return new GLiteral { Bytes = [.. bytes] };
    }

    private GNode ParseClass()
    {
        _i++; // '['
        bool negated = false;
        if (!Eof && Peek() == '^')
        {
            negated = true;
            _i++;
        }

        var cls = new GClass { Negated = negated };
        bool first = true;
        while (!Eof && !(Peek() == ']' && !first))
        {
            first = false;
            int lo = ParseEscapedCodePoint();
            if (!Eof && Peek() == '-' && _i + 1 < _s.Length && _s[_i + 1] != ']')
            {
                _i++; // '-'
                int hi = ParseEscapedCodePoint();
                if (hi < lo) throw Error($"invalid character range {lo}-{hi}");
                AddClassRange(cls, lo, hi);
            }
            else
            {
                AddClassRange(cls, lo, lo);
            }
        }
        Expect(']');
        return cls;
    }

    private void AddClassRange(GClass cls, int lo, int hi)
    {
        for (int cp = lo; cp <= hi; cp++)
        {
            if (cp < 128)
            {
                cls.Allowed[cp] = true;
                continue;
            }
            // Negated classes exclude non-ASCII code points byte-wise cannot
            // express; multi-byte exclusions are ignored (JSON grammars only
            // negate ASCII, which is exact).
            if (cls.Negated)
                continue;
            if (cls.Sequences.Count >= MaxPositiveClassSequences)
                throw Error("character class expands to too many code points");
            cls.Sequences.Add(Encoding.UTF8.GetBytes(char.ConvertFromUtf32(cp)));
        }
    }

    // ─── Trivia / tokens ───────────────────────────────────────────────────

    private void SkipTrivia()
    {
        while (!Eof)
        {
            char c = _s[_i];
            if (c == '#')
            {
                while (!Eof && _s[_i] != '\n') _i++;
            }
            else if (char.IsWhiteSpace(c))
            {
                _i++;
            }
            else
            {
                break;
            }
        }
    }

    private bool AtRuleStart()
    {
        int save = _i;
        SkipTrivia();
        bool result = false;
        if (!Eof && IsNameStart(Peek()))
        {
            ParseName();
            SkipTrivia();
            result = Match("::=");
        }
        _i = save;
        return result;
    }

    private string ParseName()
    {
        if (Eof || !IsNameStart(Peek()))
            throw Error("expected a rule name");
        int start = _i;
        while (!Eof && IsNameChar(Peek())) _i++;
        return _s[start.._i];
    }

    private int ParseNumber()
    {
        if (Eof || !char.IsDigit(Peek()))
            throw Error("expected a number");
        int value = 0;
        while (!Eof && char.IsDigit(Peek()))
        {
            value = checked(value * 10 + (_s[_i] - '0'));
            _i++;
        }
        return value;
    }

    private int ParseEscapedCodePoint()
    {
        if (Eof) throw Error("unexpected end of grammar in literal");
        char c = _s[_i++];
        if (c != '\\')
            return ReadCodePoint(ref c);

        if (Eof) throw Error("dangling escape");
        char e = _s[_i++];
        int cp = e switch
        {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'a' => 0x07,
            'b' => '\b',
            'f' => '\f',
            'v' => 0x0B,
            '0' => 0x00,
            '\\' => '\\',
            '"' => '"',
            '\'' => '\'',
            '[' => '[',
            ']' => ']',
            '(' => '(',
            ')' => ')',
            '{' => '{',
            '}' => '}',
            '|' => '|',
            '*' => '*',
            '+' => '+',
            '?' => '?',
            '-' => '-',
            '.' => '.',
            '/' => '/',
            'x' => ReadHex(2),
            'u' => ReadHex(4),
            'U' => ReadHex(8),
            _ => throw Error($"unsupported escape '\\{e}'"),
        };
        return cp;
    }

    private int ReadHex(int digits)
    {
        if (_i + digits > _s.Length) throw Error("truncated hex escape");
        int value = 0;
        for (int k = 0; k < digits; k++)
        {
            int d = HexDigit(_s[_i++]);
            if (d < 0) throw Error("invalid hex escape");
            value = (value << 4) | d;
        }
        return value;
    }

    private int ReadCodePoint(ref char c)
    {
        if (char.IsHighSurrogate(c) && _i < _s.Length && char.IsLowSurrogate(_s[_i]))
        {
            char lo = _s[_i++];
            return char.ConvertToUtf32(c, lo);
        }
        return c;
    }

    private static int HexDigit(char c)
        => c switch
        {
            >= '0' and <= '9' => c - '0',
            >= 'a' and <= 'f' => c - 'a' + 10,
            >= 'A' and <= 'F' => c - 'A' + 10,
            _ => -1,
        };

    // ─── Low-level helpers ─────────────────────────────────────────────────

    private bool Eof => _i >= _s.Length;
    private char Peek() => _s[_i];

    private void Expect(char c)
    {
        if (Eof || _s[_i] != c) throw Error($"expected '{c}'");
        _i++;
    }

    private void Expect(string text)
    {
        if (!Match(text)) throw Error($"expected '{text}'");
    }

    private bool Match(string text)
    {
        if (_i + text.Length > _s.Length) return false;
        if (!_s.AsSpan(_i, text.Length).SequenceEqual(text.AsSpan())) return false;
        _i += text.Length;
        return true;
    }

    private static bool IsNameStart(char c)
        => c == '_' || char.IsAsciiLetter(c);

    private static bool IsNameChar(char c)
        => c == '_' || c == '-' || char.IsAsciiLetterOrDigit(c);

    private GrammarException Error(string message)
        => new($"{message} (at offset {_i})");
}