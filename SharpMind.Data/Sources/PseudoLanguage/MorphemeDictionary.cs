namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class MorphemeDictionary
{
    private readonly Dictionary<string, MorphemeCategory> _prefixes;
    private readonly Dictionary<string, MorphemeCategory> _suffixes;
    private readonly List<string> _roots;

    public IReadOnlyList<string> Roots => _roots;

    public MorphemeDictionary(int targetVocabSize = 0)
    {
        _prefixes = new()
        {
            ["un"] = MorphemeCategory.Negation,
            ["non"] = MorphemeCategory.Negation,
            ["im"] = MorphemeCategory.Negation,
            ["dis"] = MorphemeCategory.Negation,
            ["anti"] = MorphemeCategory.Negation,
            ["re"] = MorphemeCategory.Repeat,
            ["pre"] = MorphemeCategory.Before,
            ["post"] = MorphemeCategory.After,
            ["over"] = MorphemeCategory.Excess,
            ["under"] = MorphemeCategory.Deficient,
            ["inter"] = MorphemeCategory.Between,
            ["trans"] = MorphemeCategory.Across,
            ["super"] = MorphemeCategory.Above,
            ["sub"] = MorphemeCategory.Below,
            ["mid"] = MorphemeCategory.Middle,
        };

        _suffixes = new()
        {
            ["er"] = MorphemeCategory.Agent,
            ["or"] = MorphemeCategory.Agent,
            ["ant"] = MorphemeCategory.Agent,
            ["ent"] = MorphemeCategory.Agent,
            ["ful"] = MorphemeCategory.Adjective,
            ["less"] = MorphemeCategory.Adjective,
            ["able"] = MorphemeCategory.Adjective,
            ["ible"] = MorphemeCategory.Adjective,
            ["ous"] = MorphemeCategory.Adjective,
            ["ive"] = MorphemeCategory.Adjective,
            ["ic"] = MorphemeCategory.Adjective,
            ["al"] = MorphemeCategory.Adjective,
            ["ed"] = MorphemeCategory.PastTense,
            ["ing"] = MorphemeCategory.PresentParticiple,
            ["ize"] = MorphemeCategory.Verb,
            ["ify"] = MorphemeCategory.Verb,
            ["en"] = MorphemeCategory.Verb,
            ["tion"] = MorphemeCategory.Noun,
            ["ment"] = MorphemeCategory.Noun,
            ["ness"] = MorphemeCategory.Noun,
            ["ity"] = MorphemeCategory.Noun,
            ["age"] = MorphemeCategory.Noun,
            ["ure"] = MorphemeCategory.Noun,
            ["s"] = MorphemeCategory.Plural,
            ["ly"] = MorphemeCategory.Adverb,
        };

        _roots =
        [
            "walk", "run", "jump", "swim", "fly", "crawl", "climb", "dance", "slide", "glide",
            "hop", "skip", "leap", "dash", "sprint",
            "speak", "talk", "say", "tell", "ask", "shout", "whisper", "call", "write", "read",
            "build", "make", "create", "form", "shape", "craft", "design", "draw", "paint", "compose",
            "find", "search", "seek", "explore", "discover", "learn", "study", "examine", "probe", "hunt",
            "start", "stop", "begin", "end", "move", "change", "turn", "shift", "drive", "push",
            "see", "look", "watch", "hear", "listen", "feel", "touch", "sense", "notice", "observe",
            "love", "hate", "fear", "hope", "wish", "dream", "laugh", "cry", "smile",
            "think", "know", "believe", "understand", "remember", "forget", "decide", "choose", "plan", "imagine",
            "do", "make", "act", "work", "play", "try", "use", "get", "give", "take",
            "be", "have", "hold", "keep", "stay", "remain", "exist", "live", "grow", "become",
            "give", "take", "get", "receive", "accept", "reject", "offer", "pass", "send", "bring",
            "help", "aid", "assist", "support", "back", "save", "rescue", "protect", "defend", "guard",
            "compare", "match", "fit", "suit", "equal", "like", "love", "prefer", "choose", "select",
            "measure", "count", "weigh", "size", "rate", "grade", "rank", "score", "assess", "value",
            "wait", "stay", "remain", "last", "endure", "continue", "persist", "age", "date",
        ];

        if (targetVocabSize > 0)
        {
            ExpandTo(targetVocabSize);
        }
    }

    private void ExpandTo(int targetSize)
    {
        if (targetSize <= _roots.Count) return;

        // Estimate roots needed
        int currentRoots = _roots.Count;
        int needed = targetSize - currentRoots;
        
        // Generate unique roots: r000, r001 ... r999, ra0 ...
        int toAdd = Math.Min(needed, 1000); // Keep adding until we have enough
        
        for (int i = 0; i < toAdd; i++)
        {
            string root = i < 100 ? $"r{i.ToString().PadLeft(3, '0')}" : $"r{i}";
            if (!_roots.Contains(root))
            {
                _roots.Add(root);
            }
        }
        
        // Add any remaining if still needed (extend format)
        while (_roots.Count < targetSize)
        {
            int idx = _roots.Count;
            string root = $"root{idx}";
            if (!_roots.Contains(root))
            {
                _roots.Add(root);
            }
        }
        
        // Ensure we have enough prefixes/suffixes too
        // (Already have ~15 each, which is likely enough for combinations)
    }

    public bool TryGetPrefixCategory(string prefix, out MorphemeCategory category)
        => _prefixes.TryGetValue(prefix, out category);

    public bool TryGetSuffixCategory(string suffix, out MorphemeCategory category)
        => _suffixes.TryGetValue(suffix, out category);

    public IEnumerable<string> GetPrefixes() => _prefixes.Keys;
    public IEnumerable<string> GetSuffixes() => _suffixes.Keys;
}

public enum MorphemeCategory
{
    Root,
    Negation,
    Repeat,
    Before,
    After,
    Excess,
    Deficient,
    Between,
    Across,
    Above,
    Below,
    Middle,
    Agent,
    Adjective,
    PastTense,
    PresentParticiple,
    Verb,
    Noun,
    Plural,
    Adverb,
}