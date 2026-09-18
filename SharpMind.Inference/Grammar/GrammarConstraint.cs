namespace SharpMind.Inference.Grammar;

/// <summary>
/// A live cursor over a compiled <see cref="GbnfGrammar"/>. It keeps the set of
/// pushdown configurations (return stack + NFA state) the input has reached and,
/// on each step, enumerates the vocabulary tokens whose bytes can extend any of
/// them. Enumerating from the token trie (rather than testing every vocabulary
/// entry against the grammar) costs time proportional to the number of
/// <em>allowed</em> tokens, which stays small for real grammars such as JSON.
///
/// Stop tokens are only ever offered when <see cref="CanStop"/> is true, so a
/// turn can never end in the middle of a structure. When the grammar is not
/// complete and no ordinary token can continue it, <see cref="IsDead"/> becomes
/// true and every logit is masked; callers should treat that as an error rather
/// than sampling an invalid token.
/// </summary>
public sealed class GrammarConstraint : IGrammarConstraint
{
    /// <summary>
    /// Logit value written for disallowed tokens. It must be finite: the sampler's
    /// softmax treats an infinite logit as corrupt input and would otherwise zero
    /// the whole distribution. At this magnitude a masked token's probability
    /// underflows to zero while a real logit still wins greedily.
    /// </summary>
    public const float MaskedLogit = -1e30f;

    private readonly GrammarNfa _nfa;
    private readonly TokenByteTable _table;
    private readonly TokenTrie _trie;
    private readonly int[] _stopIds;

    private List<Config> _state;
    private readonly List<Config> _consume;
    private List<Config> _advanceResult;
    private readonly List<Config>[] _dfsStates;
    private readonly HashSet<Config> _seen = [];
    private readonly List<Config> _workset = [];
    private readonly int[] _allowStamp;
    private readonly List<int> _allowed = [];

    private int _step;
    private int _ordinaryCount;
    private bool _canStop;

    /// <inheritdoc />
    public bool CanStop => _canStop;

    /// <inheritdoc />
    public bool IsDead { get; private set; }

    /// <summary>Number of ordinary (non-stop) tokens allowed by the last <see cref="Apply"/>.</summary>
    public int AllowedTokenCount => _ordinaryCount;

    /// <summary>
    /// True when <paramref name="tokenId"/> was left unmasked by the most recent
    /// <see cref="Apply"/>.
    /// </summary>
    public bool IsAllowed(int tokenId)
        => _step > 0
            && (uint)tokenId < (uint)_allowStamp.Length
            && _allowStamp[tokenId] == _step;

    internal GrammarConstraint(GbnfGrammar grammar, TokenByteTable table, IEnumerable<int>? stopTokenIds)
    {
        ArgumentNullException.ThrowIfNull(grammar);
        ArgumentNullException.ThrowIfNull(table);

        _nfa = grammar.Nfa;
        _table = table;
        _trie = TokenTrie.Get(table);

        int maxStop = -1;
        var stops = new List<int>();
        if (stopTokenIds is not null)
        {
            foreach (int id in stopTokenIds)
            {
                if (id < 0)
                    continue;
                maxStop = Math.Max(maxStop, id);
                stops.Add(id);
            }
        }
        _stopIds = stops.ToArray();

        _state = [.. _nfa.StartConfigs];
        _consume = [];
        _advanceResult = [];
        _dfsStates = new List<Config>[_trie.MaxDepth + 1];
        for (int i = 0; i < _dfsStates.Length; i++)
            _dfsStates[i] = [];
        _canStop = _nfa.StartCanStop;

        int stampSize = Math.Max(table.Count, maxStop + 1);
        _allowStamp = new int[stampSize];
    }

    /// <inheritdoc />
    public void Apply(Span<float> logits)
    {
        if (IsDead)
        {
            logits.Fill(MaskedLogit);
            return;
        }

        _step++;
        _allowed.Clear();
        int ordinary = 0;
        Visit(_trie.Root, _state, depth: 0, ref ordinary);
        _ordinaryCount = ordinary;

        if (_canStop)
        {
            foreach (int id in _stopIds)
            {
                if ((uint)id < (uint)_allowStamp.Length && _allowStamp[id] != _step)
                {
                    _allowStamp[id] = _step;
                    _allowed.Add(id);
                }
            }
        }

        IsDead = ordinary == 0 && !_canStop;
        if (IsDead)
        {
            logits.Fill(MaskedLogit);
            return;
        }

        int limit = Math.Min(logits.Length, _allowStamp.Length);
        for (int i = 0; i < limit; i++)
        {
            if (_allowStamp[i] != _step)
                logits[i] = MaskedLogit;
        }
        for (int i = limit; i < logits.Length; i++)
            logits[i] = MaskedLogit;
    }

    /// <inheritdoc />
    public void Accept(int tokenId)
    {
        if (IsDead || tokenId < 0 || tokenId >= _table.Count)
            return;

        ReadOnlySpan<byte> bytes = _table.GetBytes(tokenId);
        for (int i = 0; i < bytes.Length; i++)
        {
            if (!_nfa.Advance(_state, bytes[i], _consume, _advanceResult, _seen, _workset, out _canStop))
            {
                IsDead = true;
                return;
            }
            (_state, _advanceResult) = (_advanceResult, _state);
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        _state.Clear();
        _state.AddRange(_nfa.StartConfigs);
        _canStop = _nfa.StartCanStop;
        // Stamps are compared against _step, and the step counter restarts at 1,
        // so stale stamps from the previous run must be cleared or they would
        // look like tokens already allowed this step.
        Array.Clear(_allowStamp);
        IsDead = false;
        _ordinaryCount = 0;
        _allowed.Clear();
        _step = 0;
    }

    private void Visit(int node, List<Config> state, int depth, ref int ordinary)
    {
        Span<int> ids = stackalloc int[TokenTrie.MaxTokensPerPath];
        int count = _trie.CopyTokenIds(node, ids);
        for (int i = 0; i < count; i++)
        {
            int id = ids[i];
            if ((uint)id < (uint)_allowStamp.Length && _allowStamp[id] != _step)
            {
                _allowStamp[id] = _step;
                _allowed.Add(id);
                ordinary++;
            }
        }

        for (int edge = _trie.FirstEdge(node); edge >= 0; edge = _trie.NextEdge(edge))
        {
            byte b = _trie.EdgeByte(edge);
            int child = _trie.EdgeChild(edge);
            List<Config> next = _dfsStates[depth + 1];
            if (_nfa.Advance(state, b, _consume, next, _seen, _workset, out _))
                Visit(child, next, depth + 1, ref ordinary);
        }
    }
}