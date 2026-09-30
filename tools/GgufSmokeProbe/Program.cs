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

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: meta <file>   |   run <file> [prompt] [maxTokens]");
    return 2;
}

string mode = args[0];
string path = Path.GetFullPath(args[1]);
if (!File.Exists(path)) { Console.Error.WriteLine($"file not found: {path}"); return 1; }

if (mode == "meta")
    return RunMeta(path);

if (mode == "run")
    return await RunInferenceAsync(path, args);

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
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, path, LoadMode.Streaming);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig);

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
        var cts = new CancellationTokenSource();

        // StartChatAsync is an interactive loop: send one user turn, then cancel so
        // the smoke probe stops instead of blocking on a second ReadLine.
        await session.StartChatAsync(Prompt, Response, cts.Token);
        Console.WriteLine();
        Console.WriteLine("OK  generated successfully");
        return 0;

        ChatMessage Prompt()
        {
            if (promptSent) { cts.Cancel(); return new() { Content = string.Empty, Role = ChatRole.User }; }
            promptSent = true;
            return new() { Content = prompt, Role = ChatRole.User };
        }

        void Response(ChatStreamEntry entry)
        {
            if (entry.IsComplete)
            {
                cts.Cancel();
                return;
            }
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