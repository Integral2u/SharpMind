using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

// GgufSmokeProbe — manual Phase-1 smoke tool. No test runner, no CI: it exists
// to eyeball whether a real downloaded GGUF loads (or fails loudly) and, for
// generative models, that a few greedy tokens come out. Kept intentionally slim.
//
//   dotnet run --project tools/GgufSmokeProbe -- meta <file>
//   dotnet run --project tools/GgufSmokeProbe -- run  <file> ["prompt"] [maxTokens=24]
//   dotnet run --project tools/GgufSmokeProbe -- bind <file>

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: meta <file>   |   dump <file> [pattern]   |   bind <file>   |   run <file> [prompt] [maxTokens]");
    return 2;
}

string mode = args[0];
string path = Path.GetFullPath(args[1]);
if (!File.Exists(path)) { Console.Error.WriteLine($"file not found: {path}"); return 1; }

if (mode == "meta")
    return RunMeta(path);

if (mode == "dump")
    return RunDump(path, args);

if (mode == "run")
    return await RunInferenceAsync(path, args);

if (mode == "fwd")
    return RunForward(path, args);

if (mode == "load1")
    return RunLoadOne(path, args);

if (mode == "bind")
    return RunBind(path, args);

Console.Error.WriteLine($"unknown mode: {mode}");
return 2;

static int RunMeta(string path)
{
    try
    {
        var meta = GgufLoader.LoadMeta(path);
        Console.WriteLine($"OK  version={meta.Version} tensors={meta.Tensors.Count} kv={meta.KvPairs.Count}");
        foreach (var group in meta.Tensors
                     .GroupBy(t => t.Dtype)
                     .OrderBy(g => g.Key))
        {
            Console.WriteLine($"    {group.Key,-12} x{group.Count()}");
        }
        var anyBf16 = meta.Tensors.Any(t => t.Dtype == QuantDType.BF16);
        Console.WriteLine(anyBf16 ? "    -> contains BF16 tensors (functional BF16 kernels must be selected)" : "");
        return 0;
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine($"UNSUPPORTED (loud fail): {ex.Message}");
        return 3;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

static int RunDump(string path, string[] args)
{
    string pattern = args.Length > 2 ? args[2] : "";
    try
    {
        var meta = GgufLoader.LoadMeta(path);
        foreach (var kv in meta.KvPairs
                     .Select(k => $"{k.Key} = {k.Value}")
                     .OrderBy(s => s, StringComparer.Ordinal))
        {
            if (kv.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                Console.WriteLine($"KV   {kv}");
        }

        foreach (var t in meta.Tensors)
        {
            if (pattern.Length > 0 && !t.Name.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                continue;
            Console.WriteLine($"TENS {t.Name,-44} {t.Dtype,-10} [{string.Join(", ", t.Shape)}]");
        }
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

// Loads the global tensors plus a single block, which is where "Failed to load
// weights" crashes actually happen. A whole model is 45 GB for Mixtral; one block is
// ~1.4 GB, so this exercises the real read path for a fraction of the cost.
// Raw forward pass: no tokenizer, no chat template, no sampling. This is the path that
// used to throw "[router] RawQuantizedData is null..." for Qwen2-MoE, where the router
// and the shared-expert gate are F32 tensors with no raw quantized fallback.

static int RunForward(string path, string[] args)
{
    bool full = args.Any(a => a.Equals("full", StringComparison.OrdinalIgnoreCase));
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
    try
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out var _meta, out var modelConfig, out var _tok);

        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        Console.WriteLine($"mode={(full ? "Full" : "Streaming")} quantizedResident={resident} arch={modelConfig.Architecture}");

        var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        weights.InitializeWeights();
        Console.WriteLine($"weights loaded in {sw.Elapsed.TotalSeconds:F1}s");

        static string ShapeOf(SharpMind.Core.Tensors.Tensor<float>? t) =>
            t is null ? "null" : $"[{string.Join(",", t.Shape)}]={t.Shape.ElementCount}";

        var b0 = weights.Blocks[0];
        Console.WriteLine($"  blk0 IsMoE={weights.IsMoE} NumExperts={modelConfig.NumExperts} " +
            $"Hidden={modelConfig.HiddenDim} HasShared={modelConfig.HasSharedExpert}");
        Console.WriteLine($"  blk0 WRouter       = {ShapeOf(b0.WRouter)}");
        Console.WriteLine($"  blk0 SharedGateInp = {ShapeOf(b0.SharedGateInp)}");
        Console.WriteLine($"  blk0 RawRouter={(b0.RawRouter is null ? "null" : b0.RawRouter.Length.ToString())}  " +
            $"RawWgateExp={b0.RawWgateExp?.Count.ToString() ?? "null"}");

        // optimizeMemory:true (the CUI default) frees float weights that have no raw
        // quantized fallback — this is the call that used to strand the F32 router.
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
        Console.WriteLine($"  after CreateTransformer: WRouter={ShapeOf(weights.Blocks[0].WRouter)}");

        var ids = new SharpMind.Core.Tensors.Tensor<int>(1, 1);
        ids.Data[0] = 9707; // "Hello"
        sw.Restart();
        var logits = model.ForwardLastLogits(ids, null);
        Console.WriteLine($"forward OK in {sw.Elapsed.TotalSeconds:F1}s, logits[{logits.ElementCount}] " +
            $"first={logits.Data[0]:F4}");
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"forward FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

static int RunLoadOne(string path, string[] args)
{
    int layer = args.Length > 2 && int.TryParse(args[2], out int l) ? l : 0;
    // "full" and "resident" mirror the CUI: ModelCache skips Streaming by default and
    // SessionLauncher always passes quantizedResident: true, which takes a different
    // block-allocation path than the probe's defaults.
    bool full = args.Any(a => a.Equals("full", StringComparison.OrdinalIgnoreCase));
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
    try
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out var _meta, out var modelConfig, out var _tok);

        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig,
            QuantizationFactory.Create(mapping),
            path,
            full ? LoadMode.Full : LoadMode.Streaming,
            quantizedResident: resident);
        Console.WriteLine($"mode={(full ? "Full" : "Streaming")} quantizedResident={resident}");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        weights.InitializeWeights();
        Console.WriteLine($"global tensors loaded in {sw.Elapsed.TotalSeconds:F1}s");

        sw.Restart();
        if (weights is TransformerWeightsStreaming streaming)
            streaming.EnsureLayerLoadedSync(layer);
        else
            Console.WriteLine("(not a streaming weights instance; skipped the single-block read)");
        Console.WriteLine($"block {layer} loaded in {sw.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine("load OK");
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"load FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

static int RunBind(string path, string[] args)
{
    var mode = args.Length > 2 && args[2].Equals("full", StringComparison.OrdinalIgnoreCase)
        ? LoadMode.Full
        : LoadMode.Streaming;
    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    try
    {
    metaHelper.Load(path, null, out var meta, out var modelConfig, out var _tok);

    var sharpConfig = modelConfig.ForModel();
    var weights = ModelFactory.CreateWeights(
        modelConfig,
        sharpConfig,
        QuantizationFactory.Create(sharpConfig.ResolvedHardware),
        path,
        mode);
    weights.InitializeWeights();

    Console.WriteLine($"arch={meta.GetString("general.architecture")} " +
        $"MoE={weights.IsMoE} experts={modelConfig.NumExperts} topK={modelConfig.TopKExperts} " +
        $"ffn={modelConfig.FfnDim} expertFfn={modelConfig.ExpertFfnDim} " +
        $"sharedFfn={modelConfig.SharedExpertFfnDim} hasShared={modelConfig.HasSharedExpert}");

    // Walk every real tensor name through the same resolver the loader uses and report
    // which slot each one lands in. A tensor bound to a slot whose shape disagrees with
    // the file is the "raw bytes are null / wrong size" class of failure; two names
    // sharing a slot is a silent overwrite. Neither shows up in a unit test, because
    // both depend on the tensor names/order in the actual file.
    var perBlock = meta.Tensors
        .Where(t => t.Name.StartsWith("blk.", StringComparison.Ordinal))
        .ToList();
    if (perBlock.Count == 0)
    {
        Console.WriteLine("no blk.N tensors found");
        return 0;
    }

    int problems = 0;
    // Resolving a float target allocates it, and a Mixtral-scale model would need 32
    // blocks x 8 experts x 235 MB to do that for every expert, so the float-slot check
    // only runs on tensors small enough to be harmless. Quantized inference never needs
    // those duplicates anyway - it reads the raw bytes.
    const long FloatCheckLimit = 4L << 20;
    foreach (var t in perBlock)
    {
        long fileElems = 1;
        foreach (var d in t.Shape) fileElems *= d;

        var raw = weights.ResolveTarget(t.Name).rawField;
        string verdict = "ok";

        if (fileElems <= FloatCheckLimit)
        {
            var flt = weights.ResolveFloatTarget(t.Name);
            if (flt is not null && flt.Shape.ElementCount != fileElems)
            {
                verdict = $"MISMATCH float=[{string.Join(",", flt.Shape)}]={flt.Shape.ElementCount} file={fileElems}";
                problems++;
            }
        }

        // Every tensor needs a destination: a raw field or a float slot. Neither means
        // the weight is silently dropped.
        if (raw is null && fileElems > FloatCheckLimit)
        {
            verdict += (verdict == "ok" ? "" : "; ") + "no raw field and float check skipped";
            problems++;
        }
        if (verdict != "ok")
            Console.WriteLine($"  {t.Name,-44} {t.Dtype,-7} file[{string.Join(",", t.Shape)}]={fileElems} " +
                $"raw={raw ?? "-"}  {verdict}");
    }

    int blocks = meta.Tensors.Select(t => t.Name).Where(n => n.StartsWith("blk.", StringComparison.Ordinal))
        .Select(n => n[..n.IndexOf('.', 4)]).Distinct().Count();
    Console.WriteLine(problems == 0
        ? $"bind OK: {perBlock.Count} block tensors across {blocks} blocks, every slot the right size"
        : $"bind PROBLEMS: {problems}");
    return problems == 0 ? 0 : 4;
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine($"UNSUPPORTED (loud fail): {ex.Message}");
        return 3;
    }
}

static async Task<int> RunInferenceAsync(string path, string[] args)
{
    string prompt = args.Length > 2 ? args[2] : "Hello!";
    int maxTokens = args.Length > 3 && int.TryParse(args[3], out int m) ? m : 24;

    ModelMetaData meta;
    ModelConfig modelConfig;
    Tokenizer? tokenizer;
    try
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out meta, out modelConfig, out tokenizer);
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine($"UNSUPPORTED (loud fail): {ex.Message}");
        return 3;
    }

    if (tokenizer == null)
    {
        Console.WriteLine("OK  meta/config loaded; no tokenizer in file (embedding-only model?) -> verification stops here.");
        return 0;
    }

    Console.WriteLine($"OK  arch={meta.GetString("general.architecture")} tokenizer={tokenizer.VocabSize} ctx={modelConfig.MaxSeqLen}");

    try
    {
        var sharpConfig = modelConfig.ForModel();
        bool full = args.Any(a => a.Equals("full", StringComparison.OrdinalIgnoreCase));
        bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
        var mapping = sharpConfig.ToJigSawMapping();
        Console.WriteLine($"mode={(full ? "Full" : "Streaming")} quantizedResident={resident}");
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
        {
            MaxTokens = maxTokens,
            Temperature = 0f,
            TopK = 40,
            TopP = 0.95f,
        };
        session.InitializeChat();

        bool first = true;
        bool promptSent = false;
        int tokenCount = 0;
        var cts = new CancellationTokenSource();

        // StartChatAsync is an interactive loop: send one user turn, then cancel so
        // the smoke probe stops instead of blocking on a second ReadLine.
        await session.StartChatAsync(Prompt, Response, cts.Token);
        Console.WriteLine();
        Console.WriteLine($"OK  generated successfully ({tokenCount} tokens)");
        return tokenCount > 0 ? 0 : 4;

        ChatMessage Prompt()
        {
            if (promptSent) { cts.Cancel(); return new() { Content = string.Empty, Role = ChatRole.User }; }
            promptSent = true;
            return new() { Content = prompt, Role = ChatRole.User };
        }

        void Response(ChatStreamEntry entry)
        {
            if (!string.IsNullOrEmpty(entry.Error))
                Console.WriteLine($"ERR  {entry.Error}");
            if (entry.IsComplete)
            {
                cts.Cancel();
                return;
            }
            if (entry.Status == ChatStatus.Waiting || string.IsNullOrEmpty(entry.Token))
                return;
            tokenCount++;
            Console.Write(first ? "OUT: " : "");
            Console.Write(entry.Token);
            first = false;
        }
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine($"\nUNSUPPORTED (loud fail): {ex.Message}");
        return 3;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"\nFAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}