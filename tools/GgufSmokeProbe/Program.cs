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
//   dotnet run --project tools/GgufSmokeProbe -- run  <file> ["prompt"] [maxTokens=24] [full] [resident] [topk]
//   dotnet run --project tools/GgufSmokeProbe -- bind <file>
//   dotnet run --project tools/GgufSmokeProbe -- cmp  <fileA> <fileB> <tensor>
//   dotnet run --project tools/GgufSmokeProbe -- ref  <file> [prompt | @promptfile]
//
// `run ... topk` dumps per-step top-5 logits with the decoded text to stderr.
// That is the cheap way to tell a weight/indexing bug (near-uniform soup from
// step 0) apart from a cache or sampling bug (sane step 0, decay afterwards)
// without a reference implementation to diff against.
//
// `ref` runs the same forward through Reference/Qwen2MoEReference, which shares
// no code with SharpMind.Model, and prints its own top-k. When the two disagree
// the disagreement localises the bug to one of the two; when they agree, both
// are being compared against llama.cpp next. REF_BLOCKS=n limits it to the first
// n layers, REF_TRACE=1 prints per-layer magnitudes, REF_DUMP=<file> writes the
// final hidden state.

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: meta <file>   |   dump <file> [pattern]   |   bind <file>   |   fwd <file> [full] [resident]   |   fwdmulti <file> <promptfile> [single|...]   |   ref <file> [prompt]   |   run <file> [prompt] [maxTokens] [full] [resident] [topk]");
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

if (mode == "fwdmulti")
    return RunForwardMulti(path, args);

if (mode == "ref")
    return GgufSmokeProbe.Qwen2MoEReference.Run(path, args);

if (mode == "load1")
    return RunLoadOne(path, args);

if (mode == "benchload")
    return RunBenchLoad(path, args);

if (mode == "bind")
    return RunBind(path, args);

if (mode == "cmp")
    return RunCompare(path, args);

Console.Error.WriteLine($"unknown mode: {mode}");
return 2;

// `cmp <fileA> <fileB> <tensor>` dequantizes the same tensor out of two GGUF files and
// reports the Pearson correlation between them. Two quantizations of the same base model
// are lossy but strongly correlated, so a healthy dequant lands near r ~= 0.85+ regardless
// of the exact bit budget. A decode that permutes groups, mispairs scale/min nibbles or
// reads the wrong byte stride collapses r toward 0, which isolates "this format is
// decoded wrong" from "this file is a bad quant" without needing a reference build.

static int RunCompare(string pathA, string[] args)
{
    if (args.Length < 4) { Console.Error.WriteLine("cmp needs <fileA> <fileB> <tensor>"); return 2; }
    string pathB = Path.GetFullPath(args[2]);
    string tensorName = args[3];
    if (!File.Exists(pathB)) { Console.Error.WriteLine($"file not found: {pathB}"); return 1; }

    try
    {
        var metaA = GgufLoader.LoadMeta(pathA);
        var metaB = GgufLoader.LoadMeta(pathB);

        var ta = metaA.Tensors.FirstOrDefault(t => t.Name == tensorName);
        var tb = metaB.Tensors.FirstOrDefault(t => t.Name == tensorName);
        if (ta.Name is null || tb.Name is null)
        {
            Console.WriteLine($"tensor '{tensorName}' present in A={ta.Name is not null} B={tb.Name is not null}");
            return 1;
        }
        if (!ta.Shape.SequenceEqual(tb.Shape))
        {
            Console.WriteLine($"shape mismatch A=[{string.Join(",", ta.Shape)}] B=[{string.Join(",", tb.Shape)}]");
            return 1;
        }

        int n = 1;
        foreach (int d in ta.Shape) n *= d;

        var qOps = QuantizationFactory.Create((GgufLoader.LoadConfig(metaA) ?? throw new InvalidOperationException("no config")).ForModel().ToJigSawMapping());
        var a = DequantizeTensor(qOps, pathA, metaA, ta, n);
        var b = DequantizeTensor(qOps, pathB, metaB, tb, n);

        double sumA = 0, sumB = 0;
        for (int i = 0; i < n; i++) { sumA += a[i]; sumB += b[i]; }
        double meanA = sumA / n, meanB = sumB / n;

        double cov = 0, varA = 0, varB = 0, sqA = 0, sqB = 0;
        for (int i = 0; i < n; i++)
        {
            double da = a[i] - meanA, db = b[i] - meanB;
            cov += da * db; varA += da * da; varB += db * db;
            sqA += a[i] * a[i]; sqB += b[i] * b[i];
        }
        double r = cov / Math.Sqrt(varA * varB);

        Console.WriteLine($"{tensorName,-30} [{string.Join(",", ta.Shape)}]={n}  A={ta.Dtype} B={tb.Dtype}");
        Console.WriteLine($"  pearson r  = {r:F4}");
        Console.WriteLine($"  rms        A={Math.Sqrt(sqA / n):F6}  B={Math.Sqrt(sqB / n):F6}");
        Console.WriteLine($"  mean       A={meanA:F6}  B={meanB:F6}");
        Console.WriteLine("  first 8    A=[" + string.Join(", ", a.Take(8).Select(v => v.ToString("F4"))) + "]");
        Console.WriteLine("            B=[" + string.Join(", ", b.Take(8).Select(v => v.ToString("F4"))) + "]");
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"cmp FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

static float[] DequantizeTensor(QuantizationOps qOps, string path, ModelMetaData meta, TensorInfo t, int n)
{
    using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
    using var reader = new BinaryReader(fs);
    fs.Position = meta.DataOffset + t.Offset;
    var data = new float[n];
    qOps.ReadFor(t.Dtype, reader, data, n);
    return data;
}

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
        metaHelper.Load(path, null, out var _meta, out var modelConfig, out var tok);

        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        // "par" (or "par=N") fans the load out; without it the load stays
        // sequential, so two runs of the same prompt are directly comparable.
        int parDegree = 1;
        foreach (string a in args)
        {
            if (!a.StartsWith("par", StringComparison.OrdinalIgnoreCase)) continue;
            parDegree = a.Length > 3 && int.TryParse(a.AsSpan(3), out int n) ? n : 0;
        }
        Console.WriteLine($"mode={(full ? "Full" : "Streaming")} quantizedResident={resident} " +
                          $"parDegree={parDegree} arch={modelConfig.Architecture}");

        var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident,
            maxParallelLoadDegree: parDegree);
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

        if (b0.WRouter is not null)
        {
            float[] rw = b0.WRouter.Data.ToArray();
            double sum = 0, mx = 0; int zeros = 0;
            foreach (float v in rw) { sum += Math.Abs(v); if (Math.Abs(v) > mx) mx = Math.Abs(v); if (v == 0f) zeros++; }
            Console.WriteLine($"  WRouter avg|v|={sum / rw.Length:F6} max|v|={mx:F6} zeros={zeros}/{rw.Length}");
            Console.WriteLine($"  WRouter first8=[{string.Join(",", rw.Take(8).Select(v => v.ToString("F5")))}]");
        }
        void DumpPlanes(string label, Dictionary<int, byte[]>? planes, int take)
        {
            if (planes is null) { Console.WriteLine($"  {label}: null"); return; }
            Console.WriteLine($"  {label}: count={planes.Count} keys={string.Join(",", planes.Keys.OrderBy(k => k).Take(6))}");
            foreach (int e in planes.Keys.OrderBy(k => k).Take(take))
            {
                var arr = planes[e];
                double sum = 0; int mn = 255, mx = 0; long nz = 0;
                foreach (byte v in arr) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; if (v != 0) nz++; }
                Console.WriteLine($"    exp{e}: len={arr.Length} avg={sum / arr.Length:F4} min={mn} max={mx} nonzero={nz}");
            }
        }
        DumpPlanes("RawWgateExp", b0.RawWgateExp, 3);
        DumpPlanes("RawWupExp", b0.RawWupExp, 1);
        DumpPlanes("RawWdownExp", b0.RawWdownExp, 1);


        {
            float[] sg = b0.SharedGateInp.Data.ToArray();
            double sum = 0, mx = 0;
            foreach (float v in sg) { sum += Math.Abs(v); if (Math.Abs(v) > mx) mx = Math.Abs(v); }
            Console.WriteLine($"  SharedGateInp avg|v|={sum / sg.Length:F6} max|v|={mx:F6} zeros={sg.Count(v => v == 0f)}/{sg.Length}");
        }
        if (args.Any(a => a.Equals("nofwd", StringComparison.OrdinalIgnoreCase))) return 0;

        var ids = new SharpMind.Core.Tensors.Tensor<int>(1, 1);
        ids.Data[0] = 9707; // "Hello"

        // Optional "@file" prompt: run a real multi-token prefill and show what the
        // model predicts for the final position. This separates a broken forward pass
        // (wrong top token here) from a broken decode/KV-cache path (right here, wrong
        // in a real session).
        string fwdPrompt = args.Length > 2 ? args[2] : "";
        if (fwdPrompt.Length > 1 && fwdPrompt[0] == '@' && File.Exists(fwdPrompt[1..]))
            fwdPrompt = File.ReadAllText(fwdPrompt[1..]);

        if (fwdPrompt.Length > 0 && tok is not null)
        {
            int[] tids = tok.Encode(fwdPrompt, addBos: false, addEos: false);
            Console.WriteLine($"  prompt tokens={tids.Length}: {string.Join(",", tids.TakeLast(12))}");
            ids = new SharpMind.Core.Tensors.Tensor<int>(1, tids.Length);
            for (int i = 0; i < tids.Length; i++) ids.Data[i] = tids[i];
        }

        sw.Restart();
        var logits = model.ForwardLastLogits(ids, null);
        Console.WriteLine($"forward OK in {sw.Elapsed.TotalSeconds:F1}s, logits[{logits.ElementCount}] " +
            $"first={logits.Data[0]:F4}");

        if (tok is not null && logits.ElementCount > 0)
        {
            var order = new int[logits.ElementCount];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            float[] arr = logits.Data.ToArray();
            Array.Sort(order, (a, b) => arr[b].CompareTo(arr[a]));
            double mean = 0; for (int i = 0; i < arr.Length; i++) mean += arr[i];
            mean /= arr.Length;
            double var2 = 0; for (int i = 0; i < arr.Length; i++) var2 += (arr[i] - mean) * (arr[i] - mean);
            double sd = Math.Sqrt(var2 / arr.Length);
            Console.WriteLine($"  logits mean={mean:F4} std={sd:F4} top10:");
            for (int i = 0; i < 10; i++)
            {
                int id = order[i];
                string piece = tok.IdToToken(id).Replace("\n", "\\n");
                Console.WriteLine($"    {i}: {id} '{piece}' {arr[id]:F4}");
            }
            Console.WriteLine($"  margin1={arr[order[0]] - arr[order[1]]:F4}");
        }
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"forward FAILED: {ex.GetType().Name}: {ex.Message}");
        return 1;
    }
}

static int RunForwardMulti(string path, string[] args)
{
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
    bool addBos = args.Any(a => a.Equals("bos", StringComparison.OrdinalIgnoreCase));
    try
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out var _meta, out var modelConfig, out var tok);
        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            LoadMode.Full, quantizedResident: resident, maxParallelLoadDegree: 0);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        weights.InitializeWeights();
        Console.WriteLine($"loaded in {sw.Elapsed.TotalSeconds:F1}s addBos={addBos}");
        Console.WriteLine($"DIMS arch={modelConfig.Architecture} hidden={modelConfig.HiddenDim} " +
            $"headDim={modelConfig.HeadDim} heads={modelConfig.NumHeads} kvHeads={modelConfig.NumKvHeads} " +
            $"kvGroup={modelConfig.KvGroupSize} qDim={modelConfig.HiddenDim} " +
            $"keyLength={modelConfig.KeyLength} valueLength={modelConfig.ValueLength} ctx={modelConfig.MaxSeqLen}");
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        string listPath = args.Length > 2 ? args[2] : "";
        if (listPath.StartsWith("@")) listPath = listPath[1..];
        string[] prompts;
        if (args.Any(a => a.Equals("single", StringComparison.OrdinalIgnoreCase)))
        {
            prompts = new[] { File.ReadAllText(listPath) };
        }
        else
        {
            prompts = File.ReadAllLines(listPath)
                .Select(l => l.Replace("\\n", "\n").Trim())
                .Where(l => l.Length > 0).ToArray();
        }

        foreach (string p in prompts)
        {
            int[] tids = tok.Encode(p, addBos, addEos: false);
            var ids = new SharpMind.Core.Tensors.Tensor<int>(1, tids.Length);
            for (int i = 0; i < tids.Length; i++) ids.Data[i] = tids[i];
            sw.Restart();
            var logits = model.ForwardLastLogits(ids, null);
            float[] arr = logits.Data.ToArray();
            int top = 0;
            for (int i = 1; i < arr.Length; i++) if (arr[i] > arr[top]) top = i;
            int second = top == 0 ? 1 : 0;
            for (int i = 0; i < arr.Length; i++) if (i != top && arr[i] > arr[second]) second = i;
            Console.WriteLine($"PROMPT\tn={tids.Length}\tids={string.Join(",", tids)}\ttop={top}\t" +
                $"'{tok.IdToToken(top).Replace("\n", "\\n")}'\tlogit={arr[top]:F4}\t" +
                $"margin={arr[top] - arr[second]:F4}\t{sw.Elapsed.TotalSeconds:F1}s");
        }
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"fwdmulti FAILED: {ex.GetType().Name}: {ex.Message}");
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

// `benchload <file> [resident] [reps]` times a Full-mode weight load at several
// degrees of parallelism on the same file, so the fan-out can be measured
// rather than assumed. Reports median wall time and the resulting throughput;
// a degree that doesn't help shows up as a flat curve, not a regression.
static int RunBenchLoad(string path, string[] args)
{
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
    int reps = 3;
    foreach (string a in args)
        if (int.TryParse(a, out int n) && n > 0 && n < 50) reps = n;

    try
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out var _meta, out var modelConfig, out var _tok);
        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        var qOps = QuantizationFactory.Create(mapping);

        int cores = Environment.ProcessorCount;
        long fileBytes = new FileInfo(path).Length;
        Console.WriteLine($"file={Path.GetFileName(path)} {fileBytes / (1024.0 * 1024.0):F0} MiB " +
                          $"cores={cores} resident={resident} reps={reps}");

        // "degrees=1,4" restricts the sweep to those degrees (0 = auto);
        // otherwise the default ladder. Restricting it keeps a cold-cache first
        // run from being charged for degrees nobody is going to use.
        var degreesArg = args
            .FirstOrDefault(a => a.StartsWith("degrees=", StringComparison.OrdinalIgnoreCase))
            ?[("degrees=".Length)..];
        int[] degrees = degreesArg is null
            ? [1, 2, 4, 8, 0]
            : degreesArg.Split(',', StringSplitOptions.RemoveEmptyEntries)
                         .Select(int.Parse).ToArray();
        foreach (int degree in degrees)
        {
            var times = new List<double>();
            for (int r = 0; r < reps; r++)
            {
                var weights = ModelFactory.CreateWeights(
                    modelConfig, sharpConfig, qOps, path,
                    LoadMode.Full, quantizedResident: resident,
                    maxParallelLoadDegree: degree);
                var sw = System.Diagnostics.Stopwatch.StartNew();
                weights.InitializeWeights();
                sw.Stop();
                times.Add(sw.Elapsed.TotalSeconds);
                weights.Dispose();
            }
            times.Sort();
            double median = times[times.Count / 2];
            string label = degree == 0 ? "auto" : degree.ToString();
            Console.WriteLine($"  degree={label,-4} median={median,7:F2}s  " +
                              $"{fileBytes / (1024.0 * 1024.0) / median,7:F1} MiB/s");
        }
        return 0;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"benchload FAILED: {ex.GetType().Name}: {ex.Message}");
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

    // "@path" reads the prompt from a file, so multi-line few-shot prompts can be
    // compared byte-for-byte against llama.cpp without fighting shell quoting.
    if (prompt.Length > 1 && prompt[0] == '@')
    {
        string pf = prompt[1..];
        if (File.Exists(pf)) prompt = File.ReadAllText(pf);
    }

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
        bool topk = args.Any(a => a.Equals("topk", StringComparison.OrdinalIgnoreCase));

        // Per-step top-5 logits go to stderr so they interleave with the token
        // stream without polluting OUT:. The std/margin figures are what make this
        // diagnostic rather than just noise: a healthy model gives a peaked
        // distribution (large margin1), while a corrupted one gives a near-uniform
        // soup, and the two are distinguishable at step 0 without a reference.
        GeneratorDiagnostics.DumpTopLogits = topk;
        var mapping = sharpConfig.ToJigSawMapping();
        // "par" (or "par=N") fans the load out; without it the load stays
        // sequential, so two runs of the same prompt are directly comparable.
        int parDegree = 1;
        foreach (string a in args)
        {
            if (!a.StartsWith("par", StringComparison.OrdinalIgnoreCase)) continue;
            parDegree = a.Length > 3 && int.TryParse(a.AsSpan(3), out int pd) ? pd : 0;
        }
        Console.WriteLine($"mode={(full ? "Full" : "Streaming")} quantizedResident={resident} " +
                          $"parDegree={parDegree} topLogits={topk}");
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident,
            maxParallelLoadDegree: parDegree);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        // "raw" skips the chat template entirely and feeds the prompt straight to the
        // generator. Base models are frequently shipped with a chat_template they were
        // never trained for, so a templated run says nothing about the maths. "nobos"
        // toggles the leading BOS, which is the other thing llama.cpp does by default
        // and this path historically did not.
        if (args.Any(a => a.Equals("raw", StringComparison.OrdinalIgnoreCase)))
        {
            bool addBos = !args.Any(a => a.Equals("nobos", StringComparison.OrdinalIgnoreCase));
            var rawBuilder = new StandardGeneratorBuilder<KVCacherBuilder>();
            var rawGen = rawBuilder.CreateGenerator(model, tokenizer, addBos, addEos: false, caches: null);
            var rawSample = new SamplingConfig { Temperature = 0f, TopK = 1 };
            var rawGenCfg = new GenerationConfig { MaxNewTokens = maxTokens, Stream = true };
            var rawSb = new System.Text.StringBuilder();
            var rawSw = System.Diagnostics.Stopwatch.StartNew();
            await foreach (var frag in rawGen.GenerateAsync(prompt, rawSample, rawGenCfg))
                rawSb.Append(frag);
            rawSw.Stop();
            Console.WriteLine($"RAW addBos={addBos}:{rawSb}");
            Console.WriteLine($"OK  raw generated {rawSb.Length} chars in {rawSw.Elapsed.TotalSeconds:F1}s");
            GeneratorDiagnostics.DumpTopLogits = false;
            return rawSb.Length > 0 ? 0 : 4;
        }

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
        // Wall-clock over generation, so a slow model is visibly slow rather than
        // indistinguishable from a hung one.
        var genSw = System.Diagnostics.Stopwatch.StartNew();

        // StartChatAsync is an interactive loop: send one user turn, then cancel so
        // the smoke probe stops instead of blocking on a second ReadLine.
        await session.StartChatAsync(Prompt, Response, cts.Token);
        genSw.Stop();
        Console.WriteLine();
        double secs = genSw.Elapsed.TotalSeconds;
        double tps = tokenCount > 0 ? tokenCount / secs : 0;
        Console.WriteLine($"OK  generated successfully ({tokenCount} tokens in {secs:F1}s, {tps:F2} tok/s)");
        GeneratorDiagnostics.DumpTopLogits = false;
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