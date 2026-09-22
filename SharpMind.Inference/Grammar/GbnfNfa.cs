using System.Numerics;

namespace SharpMind.Inference.Grammar;

/// <summary>
/// A byte-level grammar compiled into a stack-based "recursive transition
/// network". Each rule compiles once into its own range of NFA states; entering
/// a rule is a push (a call edge records the caller's continuation on a per-use
/// stack) and completing a rule pops back to that continuation. A runtime
/// state is therefore a set of configurations (stack, state) rather than a flat
/// state bitset: the same rule can be referenced from many places and can
/// recurse without any context leaking between uses, so recursive grammars such
/// as JSON are enforced exactly.
/// </summary>
internal sealed class GrammarNfa
{
    /// <summary>
    /// Hard bound on the return-stack depth. Legitimate grammars nest far below
    /// this; it exists only to keep degenerate zero-byte recursion
    /// (<c>a ::= b</c>, <c>b ::= a</c>) from expanding forever. Pushes beyond
    /// the bound are dropped.
    /// </summary>
    public const int MaxReturnDepth = 64;

    private readonly List<(byte Byte, int Target)>[] _byteEdges;
    private readonly List<int>[] _epsilonEdges;
    private readonly List<(int Rule, int Return)>[] _ruleCalls;
    private readonly ulong[][] _closure; // within-rule epsilon closure, per state
    private readonly int _closureWords;
    private readonly int[] _stateRule;
    private readonly int[] _ruleEntry;
    private readonly int[] _ruleExit;

    /// <summary>The fully expanded initial configuration set.</summary>
    public List<Config> StartConfigs { get; }

    /// <summary>True when <see cref="StartConfigs"/> already completes the grammar.</summary>
    public bool StartCanStop { get; }

    internal GrammarNfa(
        List<(byte Byte, int Target)>[] byteEdges,
        List<int>[] epsilonEdges,
        List<(int Rule, int Return)>[] ruleCalls,
        int[] stateRule,
        int[] ruleEntry,
        int[] ruleExit,
        int startRule)
    {
        _byteEdges = byteEdges;
        _epsilonEdges = epsilonEdges;
        _ruleCalls = ruleCalls;
        _stateRule = stateRule;
        _ruleEntry = ruleEntry;
        _ruleExit = ruleExit;
        _closureWords = (byteEdges.Length + 63) >> 6;

        _closure = new ulong[byteEdges.Length][];
        for (int s = 0; s < byteEdges.Length; s++)
            _closure[s] = ComputeClosure(s);

        StartConfigs = [];
        StartCanStop = Expand(
            [new Config(ruleEntry[startRule], [])],
            StartConfigs,
            new HashSet<Config>(),
            new List<Config>());
    }

    /// <summary>
    /// Advances <paramref name="state"/> by one <paramref name="b"/>yte.
    /// <paramref name="consume"/> and <paramref name="result"/> are supplied by
    /// the caller (thread-local); on success <paramref name="result"/> holds the
    /// expanded successor configuration set and <paramref name="canStop"/>
    /// reports whether the grammar may end now. Returns false when the byte is
    /// allowed by no configuration (a dead end).
    /// </summary>
    public bool Advance(
        IReadOnlyList<Config> state, byte b,
        List<Config> consume, List<Config> result,
        HashSet<Config> seen, List<Config> workset,
        out bool canStop)
    {
        Step(state, b, StepMode.Consume, consume, null, seen, workset);
        if (consume.Count == 0)
        {
            result.Clear();
            canStop = false;
            return false;
        }
        canStop = Step(consume, b: 0, StepMode.Expand, null, result, seen, workset);
        return true;
    }

    /// <summary>
    /// Fully epsilon-expands <paramref name="raw"/> (following closures, pushes,
    /// and pops) into <paramref name="result"/>. Returns true when the expanded
    /// set reaches the root rule's exit with an empty stack.
    /// </summary>
    public bool Expand(
        IReadOnlyList<Config> raw, List<Config> result,
        HashSet<Config> seen, List<Config> workset)
        => Step(raw, b: 0, StepMode.Expand, null, result, seen, workset);

    private enum StepMode { Consume, Expand }

    /// <summary>
    /// One transformation pass over a configuration set. In <em>Consume</em>
    /// mode every configuration that can eat byte <paramref name="b"/> is copied
    /// (raw, unexpanded) to <paramref name="consume"/>; in <em>Expand</em> mode
    /// every reachable configuration is placed in <paramref name="expanded"/>.
    /// Closures, rule pushes, and rule pops are applied in both modes.
    /// </summary>
    private bool Step(
        IReadOnlyList<Config> source,
        byte b,
        StepMode mode,
        List<Config>? consume,
        List<Config>? expanded,
        HashSet<Config> seen,
        List<Config> workset)
    {
        if (mode == StepMode.Consume)
            consume!.Clear();
        else
            expanded!.Clear();

        seen.Clear();
        workset.Clear();
        foreach (Config sourceConfig in source)
            workset.Add(sourceConfig);

        bool canStop = false;
        int head = 0;
        while (head < workset.Count)
        {
            Config cfg = workset[head++];
            if (!seen.Add(cfg))
                continue;

            if (mode == StepMode.Expand)
                expanded!.Add(cfg);

            int rule = _stateRule[cfg.State];
            int ruleExit = _ruleExit[rule];
            ulong[] closure = _closure[cfg.State];
            int[] stack = cfg.Stack;

            for (int w = 0; w < _closureWords; w++)
            {
                ulong bits = closure[w];
                while (bits != 0)
                {
                    int bit = BitOperations.TrailingZeroCount(bits);
                    bits &= bits - 1;
                    int t = (w << 6) + bit;

                    if (t == ruleExit)
                    {
                        if (stack.Length == 0)
                        {
                            // The root rule is complete.
                            canStop = true;
                        }
                        else
                        {
                            // Complete a nested rule: resume the caller at its
                            // recorded continuation.
                            int[] rest = new int[stack.Length - 1];
                            if (stack.Length > 1)
                                Array.Copy(stack, rest, stack.Length - 1);
                            workset.Add(new Config(stack[stack.Length - 1], rest));
                        }
                        continue;
                    }

                    foreach ((int called, int ret) in _ruleCalls[t])
                    {
                        if (stack.Length >= MaxReturnDepth)
                            continue;
                        int[] pushed = new int[stack.Length + 1];
                        if (stack.Length > 0)
                            Array.Copy(stack, pushed, stack.Length);
                        pushed[stack.Length] = ret;
                        workset.Add(new Config(_ruleEntry[called], pushed));
                    }

                    if (mode == StepMode.Consume)
                    {
                        foreach ((byte edgeByte, int target) in _byteEdges[t])
                        {
                            if (edgeByte == b)
                                consume!.Add(new Config(target, stack));
                        }
                    }
                }
            }
        }

        return canStop;
    }

    private ulong[] ComputeClosure(int start)
    {
        var result = new ulong[_closureWords];
        result[start >> 6] |= 1UL << (start & 63);
        var stack = new Stack<int>();
        stack.Push(start);
        while (stack.Count > 0)
        {
            int s = stack.Pop();
            foreach (int t in _epsilonEdges[s])
            {
                if ((result[t >> 6] & (1UL << (t & 63))) == 0)
                {
                    result[t >> 6] |= 1UL << (t & 63);
                    stack.Push(t);
                }
            }
        }
        return result;
    }
}

/// <summary>
/// Thompson construction from the GBNF AST, producing a pushdown grammar. Rules
/// are compiled once into separate state ranges; references become call edges
/// (with a per-use return state), so recursion needs no fragment sharing.
/// </summary>
internal sealed class GbnfCompiler
{
    private const int MaxRequiredCopies = 1024;
    private const int MaxOptionalCopies = 256;

    private readonly Dictionary<string, GNode> _rules;
    private readonly Dictionary<string, int> _ruleIds;
    private readonly int[] _ruleEntry;
    private readonly int[] _ruleExit;
    private List<(byte Byte, int Target)>[] _byteEdges;
    private List<int>[] _epsilonEdges;
    private List<(int Rule, int Return)>[] _ruleCalls;
    private int[] _stateRule;
    private int _count;
    private int _currentRule;

    public GbnfCompiler(Dictionary<string, GNode> rules)
    {
        _rules = rules;
        _ruleIds = new Dictionary<string, int>(rules.Count, StringComparer.Ordinal);
        int i = 0;
        foreach (string name in rules.Keys)
            _ruleIds[name] = i++;
        _ruleEntry = new int[rules.Count];
        _ruleExit = new int[rules.Count];

        _byteEdges = new List<(byte, int)>[1];
        _epsilonEdges = new List<int>[1];
        _ruleCalls = new List<(int, int)>[1];
        _stateRule = new int[1];
        _byteEdges[0] = [];
        _epsilonEdges[0] = [];
        _ruleCalls[0] = [];
        _count = 1;

        // Every rule gets its entry/exit before any body compiles, so a
        // reference -- including a recursive one -- can always find its target.
        for (int id = 0; id < rules.Count; id++)
        {
            _currentRule = id;
            _ruleEntry[id] = NewState();
            _ruleExit[id] = NewState();
        }
    }

    public GrammarNfa Compile()
    {
        if (!_rules.ContainsKey("root"))
            throw new GrammarException("grammar must define a 'root' rule");

        foreach (var (name, body) in _rules)
        {
            _currentRule = _ruleIds[name];
            Frag frag = CompileNode(body);
            Epsilon(_ruleEntry[_currentRule], frag.Entry);
            Epsilon(frag.Exit, _ruleExit[_currentRule]);
        }

        Array.Resize(ref _byteEdges, _count);
        Array.Resize(ref _epsilonEdges, _count);
        Array.Resize(ref _ruleCalls, _count);
        Array.Resize(ref _stateRule, _count);
        return new GrammarNfa(
            _byteEdges, _epsilonEdges, _ruleCalls, _stateRule,
            _ruleEntry, _ruleExit, _ruleIds["root"]);
    }

    private readonly record struct Frag(int Entry, int Exit);

    private Frag CompileNode(GNode node) => node switch
    {
        GLiteral lit => CompileLiteral(lit.Bytes),
        GClass cls => CompileClass(cls),
        GSequence seq => CompileSequence(seq.Items),
        GAlternation alt => CompileAlternation(alt.Options),
        GReference re => CompileReference(re.Name),
        GRepeat rep => CompileRepeat(rep),
        _ => throw new GrammarException($"unsupported grammar node {node.GetType().Name}"),
    };

    private Frag CompileLiteral(byte[] bytes)
    {
        if (bytes.Length == 0)
        {
            int s = NewState();
            return new Frag(s, s);
        }
        int entry = NewState();
        int current = entry;
        foreach (byte b in bytes)
        {
            int next = NewState();
            _byteEdges[current].Add((b, next));
            current = next;
        }
        return new Frag(entry, current);
    }

    private Frag CompileClass(GClass cls)
    {
        int entry = NewState();
        int exit = NewState();

        if (!cls.Negated)
        {
            for (int b = 0; b < 256; b++)
                if (cls.Allowed[b])
                    _byteEdges[entry].Add(((byte)b, exit));
            foreach (byte[] seq in cls.Sequences)
                Epsilon(ChainAdd(entry, seq), exit);
        }
        else
        {
            for (int b = 0; b < 256; b++)
                if (!cls.Allowed[b])
                    _byteEdges[entry].Add(((byte)b, exit));
        }
        return new Frag(entry, exit);
    }

    private int ChainAdd(int from, byte[] bytes)
    {
        int current = from;
        foreach (byte b in bytes)
        {
            int next = NewState();
            _byteEdges[current].Add((b, next));
            current = next;
        }
        return current;
    }

    private Frag CompileSequence(GNode[] items)
    {
        if (items.Length == 0)
        {
            int s = NewState();
            return new Frag(s, s);
        }
        Frag result = CompileNode(items[0]);
        for (int i = 1; i < items.Length; i++)
        {
            Frag next = CompileNode(items[i]);
            Epsilon(result.Exit, next.Entry);
            result = new Frag(result.Entry, next.Exit);
        }
        return result;
    }

    private Frag CompileAlternation(GNode[] options)
    {
        if (options.Length == 0)
        {
            int s = NewState();
            return new Frag(s, s);
        }
        int entry = NewState();
        int exit = NewState();
        foreach (GNode option in options)
        {
            Frag frag = CompileNode(option);
            Epsilon(entry, frag.Entry);
            Epsilon(frag.Exit, exit);
        }
        return new Frag(entry, exit);
    }

    private Frag CompileRepeat(GRepeat repeat)
    {
        if (repeat.Max == -1)
        {
            if (repeat.Min == 0)
                return Star(repeat.Item);
            if (repeat.Min > MaxRequiredCopies)
                throw new GrammarException("repetition lower bound is too large");
            var parts = new List<Frag>(repeat.Min + 1);
            for (int k = 0; k < repeat.Min; k++)
                parts.Add(CompileNode(repeat.Item));
            parts.Add(Star(repeat.Item));
            return Concat(parts);
        }

        if (repeat.Min > MaxRequiredCopies || repeat.Max - repeat.Min > MaxOptionalCopies)
            throw new GrammarException("repetition range is too large to expand");
        var copies = new List<Frag>(repeat.Max);
        for (int k = 0; k < repeat.Min; k++)
            copies.Add(CompileNode(repeat.Item));
        for (int k = repeat.Min; k < repeat.Max; k++)
            copies.Add(Optional(repeat.Item));
        return Concat(copies);
    }

    private Frag Star(GNode item)
    {
        int entry = NewState();
        int exit = NewState();
        Frag frag = CompileNode(item);
        Epsilon(entry, frag.Entry);
        Epsilon(entry, exit);
        Epsilon(frag.Exit, entry);
        return new Frag(entry, exit);
    }

    private Frag Optional(GNode item)
    {
        int entry = NewState();
        int exit = NewState();
        Frag frag = CompileNode(item);
        Epsilon(entry, frag.Entry);
        Epsilon(frag.Exit, exit);
        Epsilon(entry, exit);
        return new Frag(entry, exit);
    }

    private Frag Concat(List<Frag> parts)
    {
        if (parts.Count == 0)
        {
            int s = NewState();
            return new Frag(s, s);
        }
        Frag result = parts[0];
        for (int i = 1; i < parts.Count; i++)
        {
            Epsilon(result.Exit, parts[i].Entry);
            result = new Frag(result.Entry, parts[i].Exit);
        }
        return result;
    }

    private Frag CompileReference(string name)
    {
        if (!_ruleIds.TryGetValue(name, out int callee))
            throw new GrammarException($"undefined rule '{name}'");
        // A reference is a call edge: entering the callee pushes the caller's
        // continuation, which is unique to this use — no shared exit to leak.
        int call = NewState();
        int cont = NewState();
        _ruleCalls[call].Add((callee, cont));
        return new Frag(call, cont);
    }

    private int NewState()
    {
        int index = _count++;
        if (index == _byteEdges.Length)
        {
            Array.Resize(ref _byteEdges, _byteEdges.Length * 2);
            Array.Resize(ref _epsilonEdges, _epsilonEdges.Length * 2);
            Array.Resize(ref _ruleCalls, _ruleCalls.Length * 2);
            Array.Resize(ref _stateRule, _stateRule.Length * 2);
        }
        _byteEdges[index] = [];
        _epsilonEdges[index] = [];
        _ruleCalls[index] = [];
        _stateRule[index] = _currentRule;
        return index;
    }

    private void Epsilon(int from, int to) => _epsilonEdges[from].Add(to);
}