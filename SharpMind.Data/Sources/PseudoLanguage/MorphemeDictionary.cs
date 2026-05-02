namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class MorphemeDictionary
{
    private readonly Dictionary<string, MorphemeCategory> _prefixes;
    private readonly Dictionary<string, MorphemeCategory> _suffixes;
    private readonly List<string> _roots;

    public IReadOnlyList<string> Roots => _roots;

    public MorphemeDictionary()
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

        _roots = new()
        {
            // Locomotion
            "walk", "run", "jump", "swim", "fly", "crawl", "climb", "dance", "slide", "glide",
            "hop", "skip", "leap", "dash", "sprint",

            // Communication
            "speak", "talk", "say", "tell", "ask", "shout", "whisper", "call", "write", "read",

            // Creation
            "build", "make", "create", "form", "shape", "craft", "design", "draw", "paint", "compose",

            // Discovery
            "find", "search", "seek", "explore", "discover", "learn", "study", "examine", "probe", "hunt",

            // Movement (abstract)
            "start", "stop", "begin", "end", "move", "change", "turn", "shift", "drive", "push",

            // Perception
            "see", "look", "watch", "hear", "listen", "feel", "touch", "sense", "notice", "observe",

            // Emotion
            "love", "hate", "fear", "hope", "wish", "dream", "feel", "laugh", "cry", "smile",

            // Cognition
            "think", "know", "believe", "understand", "remember", "forget", "decide", "choose", "plan", "imagine",

            // Action (general)
            "do", "make", "act", "work", "play", "try", "use", "get", "give", "take",

            // State
            "be", "have", "hold", "keep", "stay", "remain", "exist", "live", "grow", "become",

            // Transfer
            "give", "take", "get", "receive", "accept", "reject", "offer", "pass", "send", "bring",

            // Assistance
            "help", "aid", "assist", "support", "back", "save", "rescue", "protect", "defend", "guard",

            // Comparison
            "compare", "match", "fit", "suit", "equal", "like", "love", "prefer", "choose", "select",

            // Measurement
            "measure", "count", "weigh", "size", "rate", "grade", "rank", "score", "assess", "value",

            // Time
            "wait", "stay", "remain", "last", "endure", "continue", "persist", "last", "age", "date",
        };
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