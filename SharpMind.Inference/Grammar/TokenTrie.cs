namespace SharpMind.Inference.Grammar;

/// <summary>
/// Compact byte trie over every ordinary token's byte sequence (see
/// <see cref="TokenByteTable"/>). The runtime constraint walks it to enumerate
/// exactly the vocabulary tokens whose bytes can extend the grammar state.
///
/// Edges are stored in one flat, append-only array; each node's outgoing edges
/// form a singly linked sibling list, so any node can gain an edge at any time
/// without invalidating the others' storage. A dense 256-way node would cost
/// ~2 KB each (~1.5 GB for a 150K-token vocabulary); this layout is ~12 bytes
/// per edge plus ~20 bytes per node.
///
/// Distinct ids can share a byte sequence (a merged token whose bytes equal a
/// byte token, or a <c>&lt;0xNN&gt;</c> fallback colliding with a byte-map piece
/// in a mixed vocabulary). Every such synonym is retained: a constraint that
/// masked all but one would wrongly suppress the model's preferred spelling of
/// otherwise identical output.
/// </summary>
internal sealed class TokenTrie
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<TokenByteTable, TokenTrie> Cache = [];

    /// <summary>Returns the per-table trie, cached so repeated sessions share it.</summary>
    public static TokenTrie Get(TokenByteTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        return Cache.GetValue(table, Build);
    }

    private struct Node
    {
        public int FirstEdge;
        public int TokenId;
        public int SynonymStart;
        public int SynonymCount;
    }

    private struct Edge
    {
        public byte Byte;
        public int Child;
        public int Next;
    }

    private readonly List<Node> _nodes =
        [new Node { FirstEdge = -1, TokenId = -1, SynonymStart = -1, SynonymCount = 0 }];
    private readonly List<Edge> _edges = [];
    private readonly List<int> _synonyms = [];

    public int NodeCount => _nodes.Count;
    public int EdgeCount => _edges.Count;

    /// <summary>Number of token ids bound to the trie.</summary>
    public int TokenCount { get; private set; }

    /// <summary>Length in bytes of the longest token sequence in the trie.</summary>
    public int MaxDepth { get; private set; }

    internal int Root => 0;
    internal int FirstEdge(int node) => _nodes[node].FirstEdge;
    internal int NextEdge(int edge) => _edges[edge].Next;
    internal byte EdgeByte(int edge) => _edges[edge].Byte;
    internal int EdgeChild(int edge) => _edges[edge].Child;

    /// <summary>
    /// Writes every token id bound exactly at <paramref name="node"/> (primary
    /// first, then synonyms) into <paramref name="destination"/> and returns the
    /// number written.
    /// </summary>
    internal int CopyTokenIds(int node, Span<int> destination)
    {
        Node n = _nodes[node];
        if (n.TokenId < 0)
            return 0;

        int written = 0;
        if (written < destination.Length)
            destination[written++] = n.TokenId;
        for (int i = 0; i < n.SynonymCount && written < destination.Length; i++)
            destination[written++] = _synonyms[n.SynonymStart + i];
        return written;
    }

    /// <summary>
    /// Builds a trie from a token table. Special tokens are skipped: a grammar
    /// constrains ordinary generation tokens, and stop/special handling is
    /// decided by the generator rather than offered as a free continuation.
    /// </summary>
    public static TokenTrie Build(TokenByteTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        var trie = new TokenTrie();
        for (int id = 0; id < table.Count; id++)
        {
            if (table.IsSpecial(id))
                continue;
            ReadOnlySpan<byte> bytes = table.GetBytes(id);
            if (bytes.Length > 0)
                trie.Insert(bytes, id);
        }
        return trie;
    }

    public void Insert(ReadOnlySpan<byte> bytes, int tokenId)
    {
        if (bytes.Length == 0)
            return;

        int node = 0;
        foreach (byte b in bytes)
        {
            if (!TryGetChild(node, b, out int child))
            {
                child = _nodes.Count;
                _nodes.Add(new Node { FirstEdge = -1, TokenId = -1, SynonymStart = -1, SynonymCount = 0 });
                AddEdge(node, b, child);
            }
            node = child;
        }

        Node n = _nodes[node];
        if (n.TokenId < 0)
        {
            n.TokenId = tokenId;
            _nodes[node] = n;
        }
        else if (n.TokenId != tokenId)
        {
            if (n.SynonymCount == 0)
                n.SynonymStart = _synonyms.Count;
            n.SynonymCount++;
            _nodes[node] = n;
            _synonyms.Add(tokenId);
        }
        TokenCount++;
        if (bytes.Length > MaxDepth)
            MaxDepth = bytes.Length;
    }

    /// <summary>True when some token's byte sequence equals <paramref name="bytes"/>.</summary>
    public bool Contains(ReadOnlySpan<byte> bytes, out int tokenId)
    {
        if (TryWalk(bytes, out int node) && _nodes[node].TokenId >= 0)
        {
            tokenId = _nodes[node].TokenId;
            return true;
        }
        tokenId = -1;
        return false;
    }

    /// <summary>
    /// Writes every token id whose byte sequence equals <paramref name="bytes"/>
    /// (primary first, then any synonyms) into <paramref name="destination"/> and
    /// returns the number written. Returns 0 when no token matches. The buffer
    /// must be at least <see cref="MaxTokensPerPath"/> long to receive all ids.
    /// </summary>
    public int GetTokenIds(ReadOnlySpan<byte> bytes, Span<int> destination)
    {
        if (!TryWalk(bytes, out int node))
            return 0;

        Node n = _nodes[node];
        if (n.TokenId < 0)
            return 0;

        int written = 0;
        if (written < destination.Length)
            destination[written++] = n.TokenId;
        for (int i = 0; i < n.SynonymCount && written < destination.Length; i++)
            destination[written++] = _synonyms[n.SynonymStart + i];
        return written;
    }

    /// <summary>Upper bound on the number of ids sharing one byte sequence.</summary>
    public static int MaxTokensPerPath => 64;

    private bool TryWalk(ReadOnlySpan<byte> bytes, out int node)
    {
        node = 0;
        foreach (byte b in bytes)
        {
            if (!TryGetChild(node, b, out node))
                return false;
        }
        return true;
    }

    private bool TryGetChild(int node, byte b, out int child)
    {
        for (int e = _nodes[node].FirstEdge; e >= 0; e = _edges[e].Next)
        {
            if (_edges[e].Byte == b)
            {
                child = _edges[e].Child;
                return true;
            }
        }
        child = -1;
        return false;
    }

    private void AddEdge(int node, byte b, int child)
    {
        int edge = _edges.Count;
        _edges.Add(new Edge { Byte = b, Child = child, Next = _nodes[node].FirstEdge });
        _nodes[node] = _nodes[node] with { FirstEdge = edge };
    }
}