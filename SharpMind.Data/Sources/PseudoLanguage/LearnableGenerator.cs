namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class LearnableGenerator : IDisposable
{
    private readonly LearnableConfig _config;
    private readonly Random _random;
    private bool _disposed;

    private static readonly string[] Nouns = ["king", "queen", "dog", "cat", "bird", "fish", "child", "robot"];
    private static readonly string[] Verbs = ["eats", "sees", "loves", "hits", "runs", "jumps", "chases"];
    private static readonly string[] Objects = ["apple", "ball", "house", "tree", "fish", "book", "cake", "hat"];
    private static readonly string[] Adjectives = ["big", "small", "fast", "slow", "happy", "sad", "tall", "short"];
    private static readonly string[] Adverbs = ["quickly", "slowly", "always", "never", "often", "rarely"];
    private static readonly string[] Questions = ["what", "who", "where", "when", "why", "how"];
    private static readonly string[] Pronouns = ["does", "is", "are", "did", "can", "will"];

    private readonly Dictionary<string, int> _vocab;
    private readonly string[] _vocabWords;

    public LearnableGenerator(LearnableConfig config, Random? random = null)
    {
        _config = config;
        _random = random ?? Random.Shared;

        var allWords = new List<string>();
        
        if (config.IncludeNouns)
            allWords.AddRange(Nouns);
        if (config.IncludeVerbs)
            allWords.AddRange(Verbs);
        if (config.IncludeObjects)
            allWords.AddRange(Objects);
        if (config.IncludeAdjectives)
            allWords.AddRange(Adjectives);
        if (config.IncludeAdverbs)
            allWords.AddRange(Adverbs);
        if (config.IncludeQuestions)
            allWords.AddRange(Questions);
        if (config.IncludePronouns)
            allWords.AddRange(Pronouns);

        _vocab = new Dictionary<string, int>(allWords.Count);
        _vocabWords = [.. allWords];

        for (int i = 0; i < _vocabWords.Length; i++)
        {
            _vocab[_vocabWords[i]] = i;
        }
    }

    public IReadOnlyList<string> Vocabulary => _vocabWords;
    public int VocabSize => _vocabWords.Length;

    public LearnableSequence GenerateTrainingSample()
    {
        var pattern = _config.SyntaxPattern;

        return pattern switch
        {
            SyntaxPattern.NounVerbNoun => GenerateNounVerbNoun(),
            SyntaxPattern.AdjectiveNounVerbNoun => GenerateAdjNounVerbNoun(),
            SyntaxPattern.NounVerbAdverbNoun => GenerateNounVerbAdvNoun(),
            SyntaxPattern.AdjectiveNounVerbAdverbNoun => GenerateComplex(),
            SyntaxPattern.NounVerbNounVerbNoun => GenerateDoubleVerb(),
            SyntaxPattern.QuerySubjectVerbObject => GenerateQuerySubjectVerbObject(),
            SyntaxPattern.QuerySubjectEat => GenerateQuerySubjectEat(),
            SyntaxPattern.QuerySubjectSee => GenerateQuerySubjectSee(),
            SyntaxPattern.QuerySubjectLove => GenerateQuerySubjectLove(),
            SyntaxPattern.QuerySubjectRuns => GenerateQuerySubjectRuns(),
            SyntaxPattern.QuerySubjectHits => GenerateQuerySubjectHits(),
            SyntaxPattern.QuerySubjectChases => GenerateQuerySubjectChases(),
            SyntaxPattern.StatementSubjectVerbObject => GenerateStatementSubjectVerbObject(),
            _ => GenerateNounVerbNoun()
        };
    }

    private LearnableSequence GenerateNounVerbNoun()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string text = $"{Nouns[subjIdx]} {Verbs[verbIdx]} {Objects[objIdx]}";
        int[] ids = [_vocab[Nouns[subjIdx]], _vocab[Verbs[verbIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = Nouns[subjIdx],
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    private LearnableSequence GenerateAdjNounVerbNoun()
    {
        int adjIdx = _random.Next(Adjectives.Length);
        int nounIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string text = $"{Adjectives[adjIdx]} {Nouns[nounIdx]} {Verbs[verbIdx]} {Objects[objIdx]}";
        int[] ids = [_vocab[Adjectives[adjIdx]], _vocab[Nouns[nounIdx]], _vocab[Verbs[verbIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = $"{Adjectives[adjIdx]} {Nouns[nounIdx]}",
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[nounIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    private LearnableSequence GenerateNounVerbAdvNoun()
    {
        int nounIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int advIdx = _random.Next(Adverbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string text = $"{Nouns[nounIdx]} {Verbs[verbIdx]} {Adverbs[advIdx]} {Objects[objIdx]}";
        int[] ids = [_vocab[Nouns[nounIdx]], _vocab[Verbs[verbIdx]], _vocab[Adverbs[advIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = Nouns[nounIdx],
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[nounIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    private LearnableSequence GenerateComplex()
    {
        int adjIdx = _random.Next(Adjectives.Length);
        int nounIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int advIdx = _random.Next(Adverbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string text = $"{Adjectives[adjIdx]} {Nouns[nounIdx]} {Verbs[verbIdx]} {Adverbs[advIdx]} {Objects[objIdx]}";
        int[] ids = [_vocab[Adjectives[adjIdx]], _vocab[Nouns[nounIdx]], _vocab[Verbs[verbIdx]], 
                   _vocab[Adverbs[advIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = $"{Adjectives[adjIdx]} {Nouns[nounIdx]}",
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[nounIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    private LearnableSequence GenerateDoubleVerb()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int verb1Idx = _random.Next(Verbs.Length);
        int objIdx = _random.Next(Objects.Length);
        int verb2Idx = _random.Next(Verbs.Length);

        string text = $"{Nouns[subjIdx]} {Verbs[verb1Idx]} {Objects[objIdx]} {Verbs[verb2Idx]}";
        int[] ids = [_vocab[Nouns[subjIdx]], _vocab[Verbs[verb1Idx]], _vocab[Objects[objIdx]], _vocab[Verbs[verb2Idx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = Nouns[subjIdx],
            Verb = $"{Verbs[verb1Idx]} {Verbs[verb2Idx]}",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab[Verbs[verb1Idx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] [verb]?" → "[subject] [verb] [object]"
    private LearnableSequence GenerateQuerySubjectVerbObject()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} {Verbs[verbIdx]}";
        string answer = $"{Nouns[subjIdx]} {Verbs[verbIdx]} {Objects[objIdx]}";
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab[Verbs[verbIdx]]];
        int[] answerIds = [_vocab[Nouns[subjIdx]], _vocab[Verbs[verbIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] eat?" → "[object]"
    private LearnableSequence GenerateQuerySubjectEat()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} eat";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["eats"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "eats",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["eats"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] see?" → "[object]"
    private LearnableSequence GenerateQuerySubjectSee()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} see";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["sees"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "sees",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["sees"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] love?" → "[object]"
    private LearnableSequence GenerateQuerySubjectLove()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} love";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["loves"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "loves",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["loves"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] run?" → "[object]"
    private LearnableSequence GenerateQuerySubjectRuns()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} run";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["runs"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "runs",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["runs"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] hit?" → "[object]"
    private LearnableSequence GenerateQuerySubjectHits()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} hit";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["hits"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "hits",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["hits"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Q&A Pattern: "What does [subject] chase?" → "[object]"
    private LearnableSequence GenerateQuerySubjectChases()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int objIdx = _random.Next(Objects.Length);

        string question = $"what does {Nouns[subjIdx]} chase";
        string answer = Objects[objIdx];
        int[] questionIds = [_vocab["what"], _vocab["does"], _vocab[Nouns[subjIdx]], _vocab["chases"]];
        int[] answerIds = [_vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = question,
            TokenIds = questionIds,
            QueryText = question,
            ResponseText = answer,
            QueryTokenIds = questionIds,
            ResponseTokenIds = answerIds,
            Subject = Nouns[subjIdx],
            Verb = "chases",
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab["chases"],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    // Statement pattern: "[subject] [verb] [object]." (full sentence for causal training)
    private LearnableSequence GenerateStatementSubjectVerbObject()
    {
        int subjIdx = _random.Next(Nouns.Length);
        int verbIdx = _random.Next(Verbs.Length);
        int objIdx = _random.Next(Objects.Length);

        string text = $"{Nouns[subjIdx]} {Verbs[verbIdx]} {Objects[objIdx]}";
        int[] ids = [_vocab[Nouns[subjIdx]], _vocab[Verbs[verbIdx]], _vocab[Objects[objIdx]]];

        return new LearnableSequence
        {
            Text = text,
            TokenIds = ids,
            Subject = Nouns[subjIdx],
            Verb = Verbs[verbIdx],
            Object = Objects[objIdx],
            SubjectId = _vocab[Nouns[subjIdx]],
            VerbId = _vocab[Verbs[verbIdx]],
            ObjectId = _vocab[Objects[objIdx]],
        };
    }

    public GenerateResult GenerateBatch(int count)
    {
        var tokens = new List<int>();
        var texts = new List<string>();
        
        for (int i = 0; i < count; i++)
        {
            var sample = GenerateTrainingSample();
            tokens.AddRange(sample.TokenIds);
            texts.Add(sample.Text);
        }

        return new GenerateResult
        {
            TokenIds = [.. tokens],
            Texts = [.. texts],
            BatchSize = count
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}

public enum SyntaxPattern
{
    NounVerbNoun,
    AdjectiveNounVerbNoun,
    NounVerbAdverbNoun,
    AdjectiveNounVerbAdverbNoun,
    NounVerbNounVerbNoun,
    
    // Q&A patterns (input → expected response)
    QuerySubjectVerbObject,      // "What does X do?" → "[X] [verb] [object]"
    QuerySubjectEat,             // "What does X eat?" → "[object]"
    QuerySubjectSee,             // "What does X see?" → "[object]"
    QuerySubjectLove,            // "What does X love?" → "[object]"
    QuerySubjectRuns,            // "What does X run?" → "[object]"
    QuerySubjectHits,           // "What does X hit?" → "[object]"
    QuerySubjectChases,         // "What does X chase?" → "[object]"
    StatementSubjectVerbObject,   // "X verb object." → (same tokens for causal training)
}

public sealed class LearnableConfig
{
    public int BatchSize { get; init; } = 4;
    public int SeqLen { get; init; } = 3;
    public int TrainSamples { get; init; } = 500;
    public int TestSamples { get; init; } = 100;
    
    public SyntaxPattern SyntaxPattern { get; init; } = SyntaxPattern.NounVerbNoun;
    
    public bool IncludeNouns { get; init; } = true;
    public bool IncludeVerbs { get; init; } = true;
    public bool IncludeObjects { get; init; } = true;
    public bool IncludeAdjectives { get; init; } = false;
    public bool IncludeAdverbs { get; init; } = false;
    public bool IncludeQuestions { get; init; } = false;
    public bool IncludePronouns { get; init; } = false;

    public int ComplexityScore => 
        (IncludeNouns ? 1 : 0) + 
        (IncludeVerbs ? 1 : 0) + 
        (IncludeObjects ? 1 : 0) +
        (IncludeAdjectives ? 2 : 0) + 
        (IncludeAdverbs ? 2 : 0) +
        (IncludeQuestions ? 2 : 0) +
        (IncludePronouns ? 1 : 0) +
        (SyntaxPattern switch
        {
            SyntaxPattern.QuerySubjectVerbObject => 10,
            SyntaxPattern.QuerySubjectEat => 8,
            SyntaxPattern.QuerySubjectSee => 8,
            SyntaxPattern.QuerySubjectLove => 8,
            SyntaxPattern.QuerySubjectRuns => 8,
            SyntaxPattern.QuerySubjectHits => 8,
            SyntaxPattern.QuerySubjectChases => 8,
            SyntaxPattern.StatementSubjectVerbObject => 3,
            _ => (int)SyntaxPattern * 3
        });
}

public sealed class LearnableSequence
{
    public required string Text { get; init; }
    public required int[] TokenIds { get; init; }
    public required string Subject { get; init; }
    public required string Verb { get; init; }
    public required string Object { get; init; }
    public required int SubjectId { get; init; }
    public required int VerbId { get; init; }
    public required int ObjectId { get; init; }
    
    // Q&A fields (optional - filled for query patterns)
    public string? QueryText { get; init; }
    public string? ResponseText { get; init; }
    public int[]? QueryTokenIds { get; init; }
    public int[]? ResponseTokenIds { get; init; }
}

public sealed class GenerateResult
{
    public required int[] TokenIds { get; init; }
    public required string[] Texts { get; init; }
    public required int BatchSize { get; init; }
}