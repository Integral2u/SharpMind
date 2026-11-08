using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Text.Json.Nodes;

// GgufSmokeProbe — manual Phase-1 smoke tool. No test runner, no CI: it exists
// to eyeball whether a real downloaded GGUF loads (or fails loudly) and, for
// generative models, that a few greedy tokens come out. Kept intentionally slim.
//
//   dotnet run --project tools/GgufSmokeProbe -- meta <file>
//   dotnet run --project tools/GgufSmokeProbe -- run  <file> ["prompt"] [maxTokens=24] [full] [resident] [topk]
//   dotnet run --project tools/GgufSmokeProbe -- bench <file> ["prompt"] [warmup=16] [measure=64] [full] [resident] [serialmm] [nowide] [reps=3] [ids]
//   dotnet run --project tools/GgufSmokeProbe -- benchab <file> ["prompt"] [warmup=16] [measure=64] [full] [resident]
//   dotnet run --project tools/GgufSmokeProbe -- widebench [K=896] [N=4864] [iters=200] [chunks=0]
//   dotnet run --project tools/GgufSmokeProbe -- stages <file> ["prompt"] [warmup=16] [measure=48] [full] [resident] [serialmm] [window N]
//   dotnet run --project tools/GgufSmokeProbe -- bind <file>
//   dotnet run --project tools/GgufSmokeProbe -- cmp  <fileA> <fileB> <tensor>
//   dotnet run --project tools/GgufSmokeProbe -- ref  <file> [prompt | @promptfile]
//   dotnet run --project tools/GgufSmokeProbe -- toolprobe <file> [maxTokens=96] [prompt] [temp=0.4 | temps=0.0,0.2,0.4] [reps=1] [mode=lenient|loose|strict]
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

if (args.Length > 0 && args[0] == "gdntest")
    return GgufSmokeProbe.GdnSelfTest.Run();

if (args.Length > 1 && args[0] == "gdntestreal")
    return GgufSmokeProbe.GdnSelfTest.RunReal(Path.GetFullPath(args[1]));

if (args.Length > 1 && args[0] == "attntestreal")
    return GgufSmokeProbe.GdnSelfTest.RunAttentionReal(Path.GetFullPath(args[1]));

if (args.Length > 1 && args[0] == "attnpos")
    return GgufSmokeProbe.GdnSelfTest.RunAttentionPosition(Path.GetFullPath(args[1]));

if (args.Length > 0 && args[0] == "kbench")
    return RunKernelBench(args);

if (args.Length > 0 && args[0] == "widebench")
    return RunWideKernelBench(args);

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: meta <file>   |   dump <file> [pattern]   |   bind <file>   |   fwd <file> [full] [resident]   |   fwdmulti <file> <promptfile> [single|...]   |   ref <file> [prompt]   |   run <file> [prompt] [maxTokens] [full] [resident] [topk] [serialmm]   |   bench <file> [prompt] [warmup] [measure] [full] [resident] [serialmm] [nowide] [reps=N] [ids]   |   benchab <file> [prompt] [warmup] [measure] [full] [resident]   |   stages <file> [prompt] [warmup] [measure] [full] [resident] [serialmm]   |   widebench [K] [N] [iters]   |   kbench <q2k|q3k|q4k|q5k|q6k|q8k|q4_0|q4_1|iq4nl|q5_0|q5_1|q8_0|q8_1|f32> [K] [N] [iters] [fma|avx2|scalar]");
    return 2;
}

if (args.Length > 0 && args[0] == "mel-sine")
    return RunMelSine(args);

string mode = args[0];
string path = Path.GetFullPath(args[1]);
if (!File.Exists(path)) { Console.Error.WriteLine($"file not found: {path}"); return 1; }

if (mode == "meta")
    return RunMeta(path);

if (mode == "dump")
    return RunDump(path, args);

if (mode == "toolprobe")
    return await RunToolProbeAsync(path, args);

if (mode == "run")
    return await RunInferenceAsync(path, args);

if (mode == "bench")
    return await RunBenchAsync(path, args);

if (mode == "benchab")
    return await RunBenchABAsync(path, args);

if (mode == "stages")
    return await RunStagesAsync(path, args);

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

if (mode == "tensor")
    return RunTensorStats(path, args);

if (mode == "values")
    return RunTensorValues(path, args);

if (mode == "audio-encoder")
    return RunAudioEncoder(path, args);

if (mode == "asr")
    return await RunAsrAsync(path, args);

if (mode == "embed-check")
    return await RunEmbedCheckAsync(path, args);

if (mode == "audio-compare")
    return RunAudioCompare(path, args);

if (mode == "asr-sweep")
    return await RunAsrSweepAsync(path, args);

if (mode == "asr-ablate")
    return await RunAsrAblateAsync(path, args);

if (mode == "asr-runtime")
    return await RunAsrRuntimeAsync(path, args);

if (mode == "tts-runtime")
    return await RunTtsRuntimeAsync(path, args);

if (mode == "tts-logits")
    return await RunTtsLogitsAsync(path, args);

if (mode == "asr-files")
    return await RunAsrFilesAsync(path, args);

if (mode == "layer-importance")
    return await RunLayerImportanceAsync(path, args);

if (mode == "vl")
    return await RunVlAsync(path, args);

if (mode == "quant-compare")
    return await RunQuantCompareAsync(path, args);

if (mode == "quant-loss")
    return await RunQuantLossAsync(path, args);

if (mode == "activation-map")
    return await RunActivationMapAsync(path, args);

if (mode == "latent")
    return await RunLatentAsync(path, args);

if (mode == "prune")
    return await RunPruneAsync(path, args);

Console.Error.WriteLine($"unknown mode: {mode}");
return 2;

// `asr-runtime <backbone> <mmproj> <wav>` exercises the same Lfm3AudioRuntime the
// Studio chat service uses.
static async Task<int> RunAsrRuntimeAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: asr-runtime <backbone.gguf> <mmproj.gguf> <wav>"); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string wavPath = Path.GetFullPath(args[3]);

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    int chunkSeconds = args.Length > 4 && int.TryParse(args[4], out int cs) ? cs : 4;
    int overlapSeconds = args.Length > 5 && int.TryParse(args[5], out int os) ? os : 2;
    using var runtime = new SharpMind.Model.Audio.Lfm3AudioRuntime(model, tokenizer, config, mmprojPath)
    {
        ChunkSeconds = chunkSeconds,
        OverlapSeconds = overlapSeconds,
    };

    var watch = System.Diagnostics.Stopwatch.StartNew();
    string transcript = runtime.TranscribeWav(wavPath, "English", 32);
    watch.Stop();
    Console.WriteLine($"ASR-RUNTIME ({watch.Elapsed.TotalSeconds:F1}s): {transcript}");
    return 0;
}

// `tts-runtime <backbone> <mmproj> <text> <out.wav> [language] [voice]` exercises the native
// text-to-speech loop: MaskGIT depth decoder -> LiquidCodec -> 24 kHz WAV.
static async Task<int> RunTtsRuntimeAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 5)
    {
        Console.Error.WriteLine("usage: tts-runtime <backbone.gguf> <mmproj.gguf> <text> <out.wav> [language] [voice] [maxFrames] [seed] [temperature] [topK]");
        return 2;
    }

    string mmprojPath = Path.GetFullPath(args[2]);
    string text = args[3];
    string wavPath = Path.GetFullPath(args[4]);
    string language = args.Length > 5 ? args[5] : "American English";
    string voice = args.Length > 6 ? args[6] : "female";
    int maxFrames = args.Length > 7 && int.TryParse(args[7], out int mf) ? mf : 256;
    int? seed = args.Length > 8 && int.TryParse(args[8], out int sd) ? sd : null;
    float temperature = args.Length > 9 && float.TryParse(args[9], out float tp) ? tp : 1.0f;
    int topK = args.Length > 10 && int.TryParse(args[10], out int tk) ? tk : 128;

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    var frameWatch = System.Diagnostics.Stopwatch.StartNew();
    using var runtime = new SharpMind.Model.Audio.Lfm3TtsRuntime(model, tokenizer, config, mmprojPath)
    {
        Language = language,
        Voice = voice,
        MaxFrames = maxFrames,
        Seed = seed,
        Temperature = temperature,
        TopK = topK,
        FrameSampled = (index, codes) => Console.WriteLine(
            $"frame {index,3} ({frameWatch.Elapsed.TotalSeconds,6:F1}s): {string.Join(",", codes)}"),
        EndOfSpeech = (index, codes) => Console.WriteLine(
            $"EOS at frame {index} ({frameWatch.Elapsed.TotalSeconds:F1}s): {string.Join(",", codes)}"),
    };

    var watch = System.Diagnostics.Stopwatch.StartNew();
    float[] audio = runtime.Synthesize(text);
    watch.Stop();
    if (audio.Length == 0)
    {
        Console.WriteLine($"TTS-RUNTIME: no audio ({watch.Elapsed.TotalSeconds:F1}s)");
        return 4;
    }

    SharpMind.Model.Audio.WavFile.Write(wavPath, audio, SharpMind.Model.Audio.Lfm3TtsRuntime.SampleRate);
    double seconds = (double)audio.Length / SharpMind.Model.Audio.Lfm3TtsRuntime.SampleRate;
    Console.WriteLine(
        $"TTS-RUNTIME ({watch.Elapsed.TotalSeconds:F1}s): {seconds:F2}s, " +
        $"{audio.Length} samples, {audio.Length / 1920} frames -> {wavPath}");
    return 0;
}

// `asr-files <backbone> <mmproj> <wav>...` transcribes several WAVs with one loaded model.
static async Task<int> RunAsrFilesAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: asr-files <backbone.gguf> <mmproj.gguf> <wav>..."); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string[] wavs = args[3..];

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    using var runtime = new SharpMind.Model.Audio.Lfm3AudioRuntime(model, tokenizer, config, mmprojPath);

    foreach (string wav in wavs)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        string transcript = runtime.TranscribeWav(Path.GetFullPath(wav), "English", 64);
        Console.WriteLine($"{Path.GetFileName(wav)} ({watch.Elapsed.TotalSeconds:F1}s): {transcript}");
    }
    return 0;
}

// `prune <source> <dest> <layer,layer...>` writes a GGUF with the selected blocks removed.
static async Task<int> RunPruneAsync(string sourcePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: prune <source.gguf> <dest.gguf> <layer,layer...>"); return 2; }
    string destination = Path.GetFullPath(args[2]);
    var layers = args[3].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .Select(int.Parse).ToHashSet();

    var watch = System.Diagnostics.Stopwatch.StartNew();
    SharpMind.Model.Analysis.SurgeryExportResult result =
        SharpMind.Model.Analysis.ModelSurgeryExporter.ExportPruned(
            sourcePath, destination, layers,
            progress: (done, total) => Console.Write($"\rtensor {done}/{total}"));
    Console.WriteLine($"\nPRUNE ({watch.Elapsed.TotalSeconds:F1}s): {result.OriginalLayers} -> {result.KeptLayers} layers, " +
                      $"removed {result.RemovedLayers}, {result.Tensors} tensors, {result.Bytes / (1024.0 * 1024.0):F1} MB -> {result.Path}");
    return 0;
}

// `quant-loss <reference> <candidate> [prompt] [maxTokens=128]` measures the next-token
// cross-entropy of both files on the same calibration text (sequentially, one model at a time).
static async Task<int> RunQuantLossAsync(string referencePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 3) { Console.Error.WriteLine("usage: quant-loss <reference.gguf> <candidate.gguf> [prompt] [maxTokens]"); return 2; }
    string candidatePath = Path.GetFullPath(args[2]);
    string prompt = args.Length > 3
        ? args[3]
        : "The quick brown fox jumps over the lazy dog. A quantization should preserve the next-token distribution.";
    int maxTokens = args.Length > 4 && int.TryParse(args[4], out int mt) ? mt : 128;

    (double referenceLoss, int tokens) = Measure(referencePath, prompt, maxTokens);
    (double candidateLoss, _) = Measure(candidatePath, prompt, maxTokens);
    Console.WriteLine($"QUANT-LOSS tokens={tokens} reference={referenceLoss:F4} candidate={candidateLoss:F4} " +
                      $"delta={candidateLoss - referenceLoss:+0.0000;-0.0000;0} " +
                      $"ppl {Math.Exp(referenceLoss):F2} -> {Math.Exp(candidateLoss):F2}");
    return 0;

    static (double Loss, int Tokens) Measure(string path, string prompt, int maxTokens)
    {
        var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
        metaHelper.Load(path, null, out _, out var config, out var tokenizer);
        if (tokenizer is null) throw new InvalidOperationException("no tokenizer");
        int[] ids = tokenizer.Encode(prompt, addBos: true, addEos: false);

        var sharpConfig = config.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        using var weights = ModelFactory.CreateWeights(
            config, sharpConfig, QuantizationFactory.Create(mapping), path,
            LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
        var analyzer = new SharpMind.Model.Analysis.LayerImportanceAnalyzer(model);
        double loss = analyzer.MeasureLoss(ids, maxTokens);
        return (loss, Math.Min(ids.Length, maxTokens) - 1);
    }
}

// `latent <model> [prompt] [maxTokens=64]` projects the prompt hidden states to 2D with
// classical MDS (dual PCA) and prints the points.
static async Task<int> RunLatentAsync(string path, string[] args)
{
    await Task.CompletedTask;
    string prompt = args.Length > 2
        ? args[2]
        : "The quick brown fox jumps over the lazy dog.";
    int maxTokens = args.Length > 3 && int.TryParse(args[3], out int mt) ? mt : 64;

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(path, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), path,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

    int[] ids = tokenizer.Encode(prompt, addBos: true, addEos: false);
    var watch = System.Diagnostics.Stopwatch.StartNew();
    SharpMind.Model.Analysis.LatentSpaceMap map =
        SharpMind.Model.Analysis.LatentSpaceAnalyzer.Capture(
            model, ids, id => tokenizer.Decode([id]), maxTokens);
    watch.Stop();

    Console.WriteLine($"LATENT ({watch.Elapsed.TotalSeconds:F1}s): tokens={map.Tokens} dim={map.Dimensions} " +
                      $"explained={map.ExplainedVariance:F3}");
    foreach (SharpMind.Model.Analysis.LatentPoint point in map.Points)
        Console.WriteLine($"{point.X,9:F4} {point.Y,9:F4}  {point.TokenId,7}  {point.Text}");
    return 0;
}

// `activation-map <model> [prompt] [maxTokens=64]` prints the layers x tokens branch RMS.
static async Task<int> RunActivationMapAsync(string path, string[] args)
{
    await Task.CompletedTask;
    string prompt = args.Length > 2
        ? args[2]
        : "The quick brown fox jumps over the lazy dog.";
    int maxTokens = args.Length > 3 && int.TryParse(args[3], out int mt) ? mt : 64;

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(path, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), path,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

    int[] ids = tokenizer.Encode(prompt, addBos: true, addEos: false);
    var watch = System.Diagnostics.Stopwatch.StartNew();
    SharpMind.Model.Analysis.ActivationMap map =
        SharpMind.Model.Analysis.ActivationMapAnalyzer.Capture(model, ids, maxTokens);
    watch.Stop();

    Console.WriteLine($"ACTIVATION-MAP ({watch.Elapsed.TotalSeconds:F1}s): layers={map.Layers} tokens={map.Tokens} " +
                      $"attentionMax={map.MaxAttention:F3} ffnMax={map.MaxFeedForward:F3}");
    PrintMap("attention", map.AttentionRms, map.Layers, map.Tokens, map.MaxAttention);
    PrintMap("ffn", map.FeedForwardRms, map.Layers, map.Tokens, map.MaxFeedForward);
    return 0;

    static void PrintMap(string name, float[] values, int layers, int tokens, float max)
    {
        const string ramp = " .:-=+*#%@";
        double maxLog = Math.Log10(1 + max);
        Console.WriteLine($"{name} (log scale):");
        for (int layer = 0; layer < layers; layer++)
        {
            var line = new System.Text.StringBuilder(tokens);
            for (int token = 0; token < tokens; token++)
            {
                float value = values[layer * tokens + token];
                int index = maxLog <= 1e-9
                    ? 0
                    : (int)Math.Clamp(Math.Log10(1 + value) / maxLog * (ramp.Length - 1), 0, ramp.Length - 1);
                line.Append(ramp[index]);
            }
            Console.WriteLine($"{layer,3} {line}");
        }
    }
}

// `quant-compare <reference> <candidate> [filter] [maxTensors]` compares two quantizations
// tensor by tensor (relative L2, cosine) without a forward pass.
static async Task<int> RunQuantCompareAsync(string referencePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 3) { Console.Error.WriteLine("usage: quant-compare <reference.gguf> <candidate.gguf> [filter] [maxTensors]"); return 2; }
    string candidatePath = Path.GetFullPath(args[2]);
    string? filter = args.Length > 3 && !int.TryParse(args[3], out _) ? args[3] : null;
    int maxTensors = args.Length > 3 && int.TryParse(args[^1], out int mt) ? mt : 0;

    var watch = System.Diagnostics.Stopwatch.StartNew();
    SharpMind.Model.Analysis.QuantizationComparisonReport report =
        SharpMind.Model.Analysis.QuantizationAnalyzer.Compare(
            referencePath, candidatePath, filter, maxTensors,
            progress: (done, total) => Console.Write($"\rtensor {done}/{total}"));
    Console.WriteLine($"\n{Path.GetFileName(referencePath)} -> {Path.GetFileName(candidatePath)} ({watch.Elapsed.TotalSeconds:F1}s)");
    Console.WriteLine($"compared={report.Compared} skipped={report.Skipped} elements={report.Elements} " +
                      $"globalRelL2={report.GlobalRelativeL2:F6} meanCosine={report.MeanCosine:F6}");
    Console.WriteLine("tensor                         ref/cand        relL2     cosine    maxAbs");
    foreach (SharpMind.Model.Analysis.TensorQuantizationError tensor in report.Worst(20))
        Console.WriteLine($"{tensor.Name,-30} {tensor.ReferenceDtype}/{tensor.CandidateDtype,-6} " +
                          $"{tensor.RelativeL2,10:F6} {tensor.Cosine,10:F6} {tensor.MaxAbsError,10:F5}");
    return 0;
}

// `vl <backbone> <mmproj> <image> [prompt] [maxTokens=128]` runs the native LFM3-VL image path.
static async Task<int> RunVlAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: vl <backbone.gguf> <mmproj.gguf> <image> [prompt] [maxTokens]"); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string imagePath = Path.GetFullPath(args[3]);
    string prompt = args.Length > 4 ? args[4] : "Describe this image in one sentence.";
    int maxTokens = args.Length > 5 && int.TryParse(args[5], out int mt) ? mt : 128;

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    float[] chw = LoadImageChw(imagePath, 512, out int height, out int width);
    Console.WriteLine($"image {width}x{height}, {SharpMind.Model.Vision.Lfm3VlRuntime.ImageTokens(height, width)} vision tokens");

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    using var runtime = new SharpMind.Model.Vision.Lfm3VlRuntime(model, tokenizer, config, mmprojPath);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    string answer = runtime.Describe(chw, height, width, prompt, maxTokens);
    Console.WriteLine($"VL ({watch.Elapsed.TotalSeconds:F1}s): {answer}");
    return 0;
}

// Loads an image, resizes it to a multiple of 32 with the long side capped, and returns CHW 0..1.
static float[] LoadImageChw(string path, int maxSide, out int height, out int width)
{
    using var source = new System.Drawing.Bitmap(path);
    double scale = Math.Min(1.0, (double)maxSide / Math.Max(source.Width, source.Height));
    width = Math.Max(32, (int)Math.Round(source.Width * scale / 32.0) * 32);
    height = Math.Max(32, (int)Math.Round(source.Height * scale / 32.0) * 32);
    using var bitmap = new System.Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
    using (var graphics = System.Drawing.Graphics.FromImage(bitmap))
    {
        graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
        graphics.DrawImage(source, 0, 0, width, height);
    }

    var chw = new float[3 * height * width];
    var data = bitmap.LockBits(
        new System.Drawing.Rectangle(0, 0, width, height),
        System.Drawing.Imaging.ImageLockMode.ReadOnly,
        System.Drawing.Imaging.PixelFormat.Format32bppArgb);
    try
    {
        unsafe
        {
            for (int y = 0; y < height; y++)
            {
                byte* row = (byte*)data.Scan0 + y * data.Stride;
                for (int x = 0; x < width; x++)
                {
                    float r = (row[x * 4 + 2] / 255f - 0.5f) / 0.5f;
                    float g = (row[x * 4 + 1] / 255f - 0.5f) / 0.5f;
                    float b = (row[x * 4 + 0] / 255f - 0.5f) / 0.5f;
                    chw[y * width + x] = r;
                    chw[height * width + y * width + x] = g;
                    chw[2 * height * width + y * width + x] = b;
                }
            }
        }
    }
    finally
    {
        bitmap.UnlockBits(data);
    }
    return chw;
}

// `layer-importance <model> [prompt] [both|attn|ffn] [maxTokens=128] [maxSkip=0]` runs the
// model-agnostic layer lab: a baseline loss, one ablated pass per layer, and optionally the
// skip-layer surgery curve.
static async Task<int> RunLayerImportanceAsync(string path, string[] args)
{
    await Task.CompletedTask;
    string prompt = args.Length > 2
        ? args[2]
        : "The quick brown fox jumps over the lazy dog. Layer importance shows which transformer blocks matter most.";
    string modeText = args.Length > 3 ? args[3] : "both";
    int maxTokens = args.Length > 4 && int.TryParse(args[4], out int mt) ? mt : 128;
    int maxSkip = args.Length > 5 && int.TryParse(args[5], out int ms) ? ms : 0;
    SharpMind.Model.Analysis.LayerAblation mode = modeText.ToLowerInvariant() switch
    {
        "attn" or "attention" => SharpMind.Model.Analysis.LayerAblation.Attention,
        "ffn" or "feedforward" => SharpMind.Model.Analysis.LayerAblation.FeedForward,
        _ => SharpMind.Model.Analysis.LayerAblation.Both,
    };

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(path, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), path,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

    int[] ids = tokenizer.Encode(prompt, addBos: true, addEos: false);
    var analyzer = new SharpMind.Model.Analysis.LayerImportanceAnalyzer(model);
    var watch = System.Diagnostics.Stopwatch.StartNew();
    SharpMind.Model.Analysis.LayerImportanceReport report = analyzer.Analyze(
        ids, mode, maxTokens: maxTokens, maxSkipLayers: maxSkip,
        progress: (done, total) => Console.Write($"\rlayer {done}/{total}"));
    Console.WriteLine($"\nbaseline loss={report.BaselineLoss:F4} tokens={report.Tokens} mode={report.Mode} ({watch.Elapsed.TotalSeconds:F1}s)");
    Console.WriteLine("layer  attnRMS   ffnRMS   ablated   delta");
    foreach (SharpMind.Model.Analysis.LayerImportance layer in report.Layers)
        Console.WriteLine($"{layer.Layer,5}  {layer.AttentionRms,7:F3}  {layer.FeedForwardRms,7:F3}  {layer.AblatedLoss,8:F4}  {layer.DeltaLoss,+8:F4}");
    if (report.LeastImportant is { } least)
        Console.WriteLine($"least important: layer {least.Layer} (delta {least.DeltaLoss:+0.0000;-0.0000;0})");
    if (report.MostImportant is { } most)
        Console.WriteLine($"most important:  layer {most.Layer} (delta {most.DeltaLoss:+0.0000;-0.0000;0})");
    if (report.Surgery.Count > 0)
    {
        Console.WriteLine("surgery  layers                              loss     delta");
        foreach (SharpMind.Model.Analysis.SurgeryStep step in report.Surgery)
            Console.WriteLine($"{step.Layers.Length,7}  {string.Join(",", step.Layers),-32} {step.Loss,10:F4} {step.DeltaLoss,+8:F4}");
    }
    return 0;
}

// `tts-logits <backbone> <mmproj> <text>` prints the hidden state and the first depth pass,
// which tells a flat distribution (broken weights/hidden) from a sampling bug.
static async Task<int> RunTtsLogitsAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: tts-logits <backbone.gguf> <mmproj.gguf> <text>"); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string text = args[3];

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    using var runtime = new SharpMind.Model.Audio.Lfm3TtsRuntime(model, tokenizer, config, mmprojPath);
    Console.Write(runtime.DescribeFirstFrame(text));
    return 0;
}

// `asr-ablate <backbone> <mmproj> <wav>` runs the full 11 s prompt, then zeroes
// the first or the last half of the audio embeddings. If the output does not change,
// those tokens were not contributing to the first generated word.
static async Task<int> RunAsrAblateAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: asr-ablate <backbone> <mmproj> <wav>"); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string wavPath = Path.GetFullPath(args[3]);

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    string prefix = "<|im_start|>system\nPerform ASR in English.<|im_end|>\n<|im_start|>user\n";
    string suffix = "<|im_end|>\n<|im_start|>assistant\n";
    int[] prefixIds = tokenizer.Encode(prefix, addBos: true, addEos: false);
    int[] suffixIds = tokenizer.Encode(suffix, addBos: false, addEos: false);
    const int audioId = 125031;

    float[] samples = SharpMind.Model.Audio.WavFile.ReadMono(wavPath, 16000);
    float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(samples);
    using var reader = new GgufTensorReader(mmprojPath);
    using var encoder = new SharpMind.Model.Audio.Lfm3AudioEncoder(reader);
    float[,] audio = encoder.Encode(mel);
    int audioCount = audio.GetLength(0);

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    int hidden = config.HiddenDim;

    var ids = new List<int>(prefixIds.Length + audioCount + suffixIds.Length);
    ids.AddRange(prefixIds);
    for (int i = 0; i < audioCount; i++) ids.Add(audioId);
    ids.AddRange(suffixIds);

    foreach (string variant in new[] { "full", "zerofirst", "zerolast" })
    {
        var idTensor = new Tensor<int>(1, ids.Count);
        for (int i = 0; i < ids.Count; i++) idTensor.Data[i] = ids[i];
        Tensor<float> embeddings = model.ForwardEmbedding(idTensor);
        idTensor.Dispose();
        int start = prefixIds.Length;
        for (int t = 0; t < audioCount; t++)
        {
            bool zero = variant == "zerofirst" ? t < audioCount / 2
                : variant == "zerolast" ? t >= audioCount / 2
                : false;
            for (int d = 0; d < hidden; d++)
                embeddings.Data[(start + t) * hidden + d] = zero ? 0 : audio[t, d];
        }

        var caches = CreateCaches(config, 2048);
        Tensor<float> logits = model.ForwardLastLogitsEmbedding(embeddings, caches, 0);
        int position = ids.Count;
        var generated = new List<int>();
        int next = ArgMax(logits.Data[..config.VocabSize]);
        logits.Dispose();
        for (int i = 0; i < 4 && next != tokenizer.EosId; i++)
        {
            generated.Add(next);
            var token = new Tensor<int>(1, 1);
            token.Data[0] = next;
            logits = model.ForwardLastLogits(token, caches, position++);
            token.Dispose();
            next = ArgMax(logits.Data[..config.VocabSize]);
            logits.Dispose();
        }
        Console.WriteLine($"{variant,-10} text='{tokenizer.Decode(generated.ToArray())}'");
    }
    return 0;
}

// `asr-sweep <backbone> <mmproj> <wav> [language] [lengths]` loads the model once and
// decodes the same audio prefix at several audio-token lengths, to find a length cutoff.
static async Task<int> RunAsrSweepAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4) { Console.Error.WriteLine("usage: asr-sweep <backbone> <mmproj> <wav> [language] [lengths]"); return 2; }
    string mmprojPath = Path.GetFullPath(args[2]);
    string wavPath = Path.GetFullPath(args[3]);
    string language = args.Length > 4 ? args[4] : "English";
    int[] lengths = args.Length > 5
        ? [.. args[5].Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)]
        : [50, 90, 138];

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    string prefix = $"<|im_start|>system\nPerform ASR in {language}.<|im_end|>\n<|im_start|>user\n";
    string suffix = "<|im_end|>\n<|im_start|>assistant\n";
    int[] prefixIds = tokenizer.Encode(prefix, addBos: true, addEos: false);
    int[] suffixIds = tokenizer.Encode(suffix, addBos: false, addEos: false);
    int audioId = 125031;

    float[] samples = SharpMind.Model.Audio.WavFile.ReadMono(wavPath, 16000);
    float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(samples);
    using var reader = new GgufTensorReader(mmprojPath);
    using var encoder = new SharpMind.Model.Audio.Lfm3AudioEncoder(reader);
    float[,] audio = encoder.Encode(mel);
    int audioCount = audio.GetLength(0);
    Console.WriteLine($"audio tokens={audioCount}");

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
    int hidden = config.HiddenDim;

    foreach (int requested in lengths)
    {
        int count = Math.Min(requested, audioCount);
        var ids = new List<int>(prefixIds.Length + count + suffixIds.Length);
        ids.AddRange(prefixIds);
        for (int i = 0; i < count; i++) ids.Add(audioId);
        ids.AddRange(suffixIds);

        var idTensor = new Tensor<int>(1, ids.Count);
        for (int i = 0; i < ids.Count; i++) idTensor.Data[i] = ids[i];
        Tensor<float> embeddings = model.ForwardEmbedding(idTensor);
        idTensor.Dispose();
        int start = prefixIds.Length;
        for (int t = 0; t < count; t++)
            for (int d = 0; d < hidden; d++)
                embeddings.Data[(start + t) * hidden + d] = audio[t, d];

        var caches = CreateCaches(config, Math.Max(2048, ids.Count + 8));
        Tensor<float> logits = model.ForwardLastLogitsEmbedding(embeddings, caches, 0);
        int position = ids.Count;
        var generated = new List<int>();
        int next = ArgMax(logits.Data[..config.VocabSize]);
        logits.Dispose();
        for (int i = 0; i < 4 && next != tokenizer.EosId; i++)
        {
            generated.Add(next);
            var token = new Tensor<int>(1, 1);
            token.Data[0] = next;
            logits = model.ForwardLastLogits(token, caches, position++);
            token.Dispose();
            next = ArgMax(logits.Data[..config.VocabSize]);
            logits.Dispose();
        }
        Console.WriteLine($"audioTokens={count} prompt={ids.Count} text='{tokenizer.Decode(generated.ToArray())}'");
    }
    return 0;
}

// `audio-compare <mmproj> <short.wav> <long.wav>` encodes both clips and compares the
// overlapping tokens. A length-dependent encoder bug shows up as a large difference even
// though the audio prefix is identical.
static int RunAudioCompare(string path, string[] args)
{
    if (args.Length < 4) { Console.Error.WriteLine("usage: audio-compare <mmproj> <short.wav> <long.wav>"); return 2; }
    using var reader = new SharpMind.Model.Format.GgufTensorReader(path);
    using var encoder = new SharpMind.Model.Audio.Lfm3AudioEncoder(reader);

    float[,] EncodeClip(string wav)
    {
        float[] samples = SharpMind.Model.Audio.WavFile.ReadMono(wav, 16000);
        float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(samples);
        return encoder.Encode(mel);
    }

    float[,] shortEmb = EncodeClip(Path.GetFullPath(args[2]));
    float[,] longEmb = EncodeClip(Path.GetFullPath(args[3]));
    int tokens = Math.Min(shortEmb.GetLength(0), longEmb.GetLength(0));
    int dim = shortEmb.GetLength(1);
    double sumDiff = 0, sumShort = 0, sumLong = 0;
    float maxDiff = 0;
    for (int t = 0; t < tokens; t++)
    {
        for (int d = 0; d < dim; d++)
        {
            double diff = Math.Abs(shortEmb[t, d] - longEmb[t, d]);
            sumDiff += diff * diff;
            sumShort += shortEmb[t, d] * shortEmb[t, d];
            sumLong += longEmb[t, d] * longEmb[t, d];
            if (diff > maxDiff) maxDiff = (float)diff;
        }
    }

    Console.WriteLine(
        $"short={shortEmb.GetLength(0)} long={longEmb.GetLength(0)} shared={tokens} " +
        $"rmsDiff={Math.Sqrt(sumDiff / Math.Max(1, (long)tokens * dim)):G6} " +
        $"maxDiff={maxDiff:G6} shortRms={Math.Sqrt(sumShort / Math.Max(1, (long)tokens * dim)):G6} " +
        $"longRms={Math.Sqrt(sumLong / Math.Max(1, (long)tokens * dim)):G6}");
    return 0;
}

// `embed-check <model> ["prompt"]` runs the same prompt through the token-id path and the
// precomputed-embedding path. They must produce the same next-token logits; this isolates
// the embedding forward from the audio encoder.
static async Task<int> RunEmbedCheckAsync(string path, string[] args)
{
    await Task.CompletedTask;
    string prompt = args.Length > 2 ? args[2] : "What is 2+2? Answer briefly.";

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(path, null, out _, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer"); return 3; }

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), path,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

    int[] ids = tokenizer.Encode(prompt, addBos: true, addEos: false);
    var idTensor = new Tensor<int>(1, ids.Length);
    for (int i = 0; i < ids.Length; i++) idTensor.Data[i] = ids[i];

    int maxSeq = ids.Length + 8;
    Tensor<float> direct = model.ForwardLastLogits(idTensor, CreateCaches(config, Math.Max(2048, maxSeq)), 0);
    Tensor<float> embeddings = model.ForwardEmbedding(idTensor);
    Tensor<float> viaEmbeds = model.ForwardLastLogitsEmbedding(embeddings, CreateCaches(config, Math.Max(2048, maxSeq)), 0);
    idTensor.Dispose();

    int vocab = config.VocabSize;
    int bestDirect = ArgMax(direct.Data[..vocab]);
    int bestEmbeds = ArgMax(viaEmbeds.Data[..vocab]);
    float maxDiff = 0;
    for (int i = 0; i < vocab; i++)
        maxDiff = Math.Max(maxDiff, Math.Abs(direct.Data[i] - viaEmbeds.Data[i]));

    Console.WriteLine($"ids={ids.Length} directArgMax={bestDirect} embedArgMax={bestEmbeds} maxDiff={maxDiff:G6}");
    Console.WriteLine($"direct='{tokenizer.Decode([bestDirect])}' embed='{tokenizer.Decode([bestEmbeds])}'");
    direct.Dispose();
    viaEmbeds.Dispose();
    return bestDirect == bestEmbeds && maxDiff < 1e-3f ? 0 : 4;
}

static IKVCache[] CreateCaches(ModelConfig config, int maxSeq)
{
    var cacheBuilder = new KVCacherBuilder();
    var caches = new IKVCache[config.NumLayers];
    for (int layer = 0; layer < config.NumLayers; layer++)
    {
        caches[layer] = config.IsShortConvLayer(layer)
            ? new ShortConvCache(config.ShortConvCacheLength - 1, config.HiddenDim, 1, maxSeq)
            : config.IsRecurrentLayer(layer)
                ? new DeltaNetCache(config.SsmConvKernel, config.SsmConvDim, config.SsmTimeStepRank,
                    config.SsmStateSize, config.SsmValueHeadDim, 1, maxSeq)
                : cacheBuilder.CreateKVCache(1, config.NumKvHeads, maxSeq, config.HeadDimForLayer(layer));
    }
    return caches;
}

// `mel-sine` prints the LFM3-Audio log-mel for a 440 Hz sine, matching the synthetic
// input llama-mtmd-debug uses for `-p preproc --audio 440 -n 16000`.
static int RunMelSine(string[] args)
{
    int samples = args.Length > 1 && int.TryParse(args[1], out int n) ? n : 16000;
    int frequency = args.Length > 2 && int.TryParse(args[2], out int f) ? f : 440;
    var waveform = new float[samples];
    for (int i = 0; i < samples; i++)
        waveform[i] = (float)Math.Sin(2 * Math.PI * frequency * i / 16000.0);

    float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(waveform);
    Console.WriteLine($"samples={samples} frames={mel.GetLength(0)} mel={mel.GetLength(1)}");
    foreach (int m in new[] { 0, 10, 18, 20, 25, 64, 100, 127 })
        Console.WriteLine($"mel[0][m={m}][t=50] = {mel[50, m]:F6}");
    return 0;
}

// `asr <backbone.gguf> <mmproj.gguf> <wav> [language] [maxTokens]` runs the native
// LFM3-Audio speech-to-text path: encode the WAV, place the audio embeddings at the
// <|audio|> position, and decode greedily.
static async Task<int> RunAsrAsync(string backbonePath, string[] args)
{
    await Task.CompletedTask;
    if (args.Length < 4)
    {
        Console.Error.WriteLine("usage: asr <backbone.gguf> <mmproj.gguf> <audio.wav> [language] [maxTokens]");
        return 2;
    }

    string mmprojPath = Path.GetFullPath(args[2]);
    string wavPath = Path.GetFullPath(args[3]);
    string language = args.Length > 4 ? args[4] : "English";
    int maxTokens = args.Length > 5 && int.TryParse(args[5], out int parsed) ? parsed : 96;
    if (!File.Exists(mmprojPath)) { Console.Error.WriteLine($"file not found: {mmprojPath}"); return 1; }
    if (!File.Exists(wavPath)) { Console.Error.WriteLine($"file not found: {wavPath}"); return 1; }

    var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor(ModelFormat.Gguf);
    metaHelper.Load(backbonePath, null, out var meta, out var config, out var tokenizer);
    if (tokenizer is null) { Console.Error.WriteLine("no tokenizer in the backbone"); return 3; }

    string prefix = $"<|im_start|>system\nPerform ASR in {language}.<|im_end|>\n<|im_start|>user\n";
    string suffix = "<|im_end|>\n<|im_start|>assistant\n";
    int[] prefixIds = tokenizer.Encode(prefix, addBos: true, addEos: false);
    int[] suffixIds = tokenizer.Encode(suffix, addBos: false, addEos: false);
    int audioId = 125031;
    try
    {
        int id = tokenizer.TokenToId("<|audio|>");
        if (id >= 0) audioId = id;
    }
    catch
    {
        // Fall back to the LFM3-Audio config value.
    }

    float[] samples = SharpMind.Model.Audio.WavFile.ReadMono(wavPath, 16000);
    float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(samples);
    using var reader = new GgufTensorReader(mmprojPath);
    using var encoder = new SharpMind.Model.Audio.Lfm3AudioEncoder(reader);
    var encodeWatch = System.Diagnostics.Stopwatch.StartNew();
    float[,] audio = encoder.Encode(mel);
    encodeWatch.Stop();
    int audioCount = audio.GetLength(0);

    var ids = new List<int>(prefixIds.Length + audioCount + suffixIds.Length);
    ids.AddRange(prefixIds);
    for (int i = 0; i < audioCount; i++) ids.Add(audioId);
    ids.AddRange(suffixIds);
    Console.WriteLine(
        $"audio {samples.Length / 16000.0:F2}s mel={mel.GetLength(0)} tokens={audioCount} " +
        $"encode={encodeWatch.Elapsed.TotalSeconds:F1}s prompt={ids.Count} audioId={audioId} " +
        $"prefix={prefixIds.Length} suffix={suffixIds.Length}");
    Console.WriteLine($"prefix='{tokenizer.Decode(prefixIds)}' suffix='{tokenizer.Decode(suffixIds)}'");

    var sharpConfig = config.ForModel();
    var mapping = sharpConfig.ToJigSawMapping();
    using var weights = ModelFactory.CreateWeights(
        config, sharpConfig, QuantizationFactory.Create(mapping), backbonePath,
        LoadMode.Full, quantizedResident: false, maxParallelLoadDegree: 1);
    weights.InitializeWeights();
    using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

    var idTensor = new Tensor<int>(1, ids.Count);
    for (int i = 0; i < ids.Count; i++) idTensor.Data[i] = ids[i];
    Tensor<float> embeddings = model.ForwardEmbedding(idTensor);
    idTensor.Dispose();

    int hidden = embeddings.Shape[2];
    int audioStart = prefixIds.Length;
    var embeddingData = embeddings.Data;
    for (int t = 0; t < audioCount; t++)
        for (int d = 0; d < hidden; d++)
            embeddingData[(audioStart + t) * hidden + d] = audio[t, d];

    var cacheBuilder = new KVCacherBuilder();
    int maxSeq = ids.Count + maxTokens + 8;
    var caches = new IKVCache[config.NumLayers];
    for (int layer = 0; layer < config.NumLayers; layer++)
    {
        caches[layer] = config.IsShortConvLayer(layer)
            ? new ShortConvCache(config.ShortConvCacheLength - 1, config.HiddenDim, 1, maxSeq)
            : config.IsRecurrentLayer(layer)
                ? new DeltaNetCache(config.SsmConvKernel, config.SsmConvDim, config.SsmTimeStepRank,
                    config.SsmStateSize, config.SsmValueHeadDim, 1, maxSeq)
                : cacheBuilder.CreateKVCache(1, config.NumKvHeads, maxSeq, config.HeadDimForLayer(layer));
    }

    var decodeWatch = System.Diagnostics.Stopwatch.StartNew();
    Tensor<float> logits = model.ForwardLastLogitsEmbedding(embeddings, caches, 0);
    int position = ids.Count;
    var generated = new List<int>();
    int next = ArgMax(logits.Data[..config.VocabSize]);
    logits.Dispose();

    while (generated.Count < maxTokens && next != tokenizer.EosId)
    {
        generated.Add(next);
        var token = new Tensor<int>(1, 1);
        token.Data[0] = next;
        logits = model.ForwardLastLogits(token, caches, position++);
        token.Dispose();
        next = ArgMax(logits.Data[..config.VocabSize]);
        logits.Dispose();
    }
    decodeWatch.Stop();

    string transcript = tokenizer.Decode(generated.ToArray());
    Console.WriteLine($"ASR ({generated.Count} tokens, {decodeWatch.Elapsed.TotalSeconds:F1}s): {transcript}");
    return 0;
}

static int ArgMax(ReadOnlySpan<float> values)
{
    int best = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

// `audio-encoder <mmproj> <wav>` runs the native LFM3-Audio speech encoder and prints the
// embedding shape and value statistics. This validates the encoder path before the ASR
// prompt wiring lands.
static int RunAudioEncoder(string path, string[] args)
{
    if (args.Length < 3)
    {
        Console.Error.WriteLine("usage: audio-encoder <mmproj.gguf> <audio.wav>");
        return 2;
    }

    string wav = Path.GetFullPath(args[2]);
    if (!File.Exists(wav)) { Console.Error.WriteLine($"file not found: {wav}"); return 1; }

    float[] samples = SharpMind.Model.Audio.WavFile.ReadMono(wav, 16000);
    float[,] mel = SharpMind.Model.Audio.LogMelSpectrogram.Compute(samples);
    Console.WriteLine($"audio {samples.Length / 16000.0:F2}s samples={samples.Length} mel={mel.GetLength(0)}x{mel.GetLength(1)}");

    using var reader = new SharpMind.Model.Format.GgufTensorReader(path);
    using var encoder = new SharpMind.Model.Audio.Lfm3AudioEncoder(reader);
    var watch = System.Diagnostics.Stopwatch.StartNew();
    float[,] embeddings = encoder.Encode(mel);
    watch.Stop();

    double sum = 0;
    float min = float.PositiveInfinity, max = float.NegativeInfinity;
    int bad = 0;
    for (int t = 0; t < embeddings.GetLength(0); t++)
    {
        for (int d = 0; d < embeddings.GetLength(1); d++)
        {
            float value = embeddings[t, d];
            if (!float.IsFinite(value)) { bad++; continue; }
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
        }
    }
    long count = (long)embeddings.GetLength(0) * embeddings.GetLength(1);
    Console.WriteLine(
        $"tokens={embeddings.GetLength(0)} dim={embeddings.GetLength(1)} " +
        $"min={min:G6} max={max:G6} mean={(count == 0 ? 0 : sum / count):G6} " +
        $"nonfinite={bad} time={watch.Elapsed.TotalSeconds:F1}s");

    int tokens = embeddings.GetLength(0);
    foreach (int t in new[] { 0, tokens / 8, tokens / 4, tokens / 2, 3 * tokens / 4, tokens - 1 })
    {
        if (t < 0 || t >= tokens) continue;
        double norm = 0;
        for (int d = 0; d < embeddings.GetLength(1); d++)
            norm += embeddings[t, d] * embeddings[t, d];
        Console.WriteLine($"token[{t}] norm={Math.Sqrt(norm):G6}");
    }
    return 0;
}

// `tensor <file> [name...]` prints dtype, shape and value statistics. With no names it
// lists the first few tensors, which is enough to inventory an mmproj or codec file.
// `values <file> <tensor> [count]` prints the first values, for checking a table's layout.
static int RunTensorValues(string path, string[] args)
{
    if (args.Length < 3) { Console.Error.WriteLine("values needs <file> <tensor> [count]"); return 2; }
    string name = args[2];
    int count = args.Length > 3 && int.TryParse(args[3], out int c) ? c : 16;
    using var reader = new SharpMind.Model.Format.GgufTensorReader(path);
    if (!reader.TryGetInfo(name, out SharpMind.Model.Format.TensorInfo info))
    {
        Console.WriteLine($"missing {name}");
        return 1;
    }
    float[] data = reader.ReadFloat32(name);
    Console.WriteLine($"{name} {info.Dtype} [{string.Join(",", info.Shape)}]");
    for (int i = 0; i < Math.Min(count, data.Length); i++)
        Console.WriteLine($"[{i}] = {data[i]:G6}");
    return 0;
}

static int RunTensorStats(string path, string[] args)
{
    using var reader = new SharpMind.Model.Format.GgufTensorReader(path);
    string[] names = args.Length > 2 ? args[2..] : [.. reader.TensorNames.Take(8)];
    foreach (string name in names)
    {
        if (!reader.TryGetInfo(name, out SharpMind.Model.Format.TensorInfo info))
        {
            Console.WriteLine($"missing {name}");
            continue;
        }

        float[] data = reader.ReadFloat32(name);
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        double sum = 0;
        int bad = 0;
        foreach (float value in data)
        {
            if (!float.IsFinite(value)) { bad++; continue; }
            if (value < min) min = value;
            if (value > max) max = value;
            sum += value;
        }
        double mean = data.Length == 0 ? 0 : sum / data.Length;
        Console.WriteLine(
            $"{name} {info.Dtype} [{string.Join(",", info.Shape)}] " +
            $"n={data.Length} min={min:G6} max={max:G6} mean={mean:G6} nonfinite={bad}");
    }
    return 0;
}

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

// `toolprobe <file> [maxTokens=96] [prompt] [temp=0.4 | temps=0.0,0.2,0.4] [reps=1] [mode=lenient|loose|strict]`
// answers one question cheaply: does
// this model, through the prompt formatter and the detected tool-call format,
// emit a call SharpMind can parse? It registers a single present_choices tool,
// sends one prompt, and prints the resolved format, the raw response, and the
// parsed call. Nothing here is game-specific.
static async Task<int> RunToolProbeAsync(string path, string[] args)
{
    int maxTokens = 96;
    string? prompt = null;
    var temperatures = new List<float> { 0.4f };
    int reps = 1;
    int positional = 0;
    ToolCallRecoveryMode recovery = ToolCallRecoveryMode.Lenient;
    for (int i = 2; i < args.Length; i++)
    {
        string a = args[i];
        if (a.StartsWith("temps=", StringComparison.OrdinalIgnoreCase))
        {
            var parsedTemps = a[6..]
                .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => float.TryParse(s, System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float v) ? v : float.NaN)
                .Where(v => !float.IsNaN(v))
                .ToList();
            if (parsedTemps.Count > 0)
                temperatures = parsedTemps;
        }
        else if (a.StartsWith("temp=", StringComparison.OrdinalIgnoreCase) &&
            float.TryParse(a[5..], System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out float t))
        {
            temperatures = [t];
        }
        else if (a.StartsWith("reps=", StringComparison.OrdinalIgnoreCase) &&
                 int.TryParse(a[5..], out int r) && r > 0)
        {
            reps = r;
        }
        else if (a.StartsWith("mode=", StringComparison.OrdinalIgnoreCase) &&
                 Enum.TryParse<ToolCallRecoveryMode>(a[5..], true, out var parsedRecovery))
        {
            recovery = parsedRecovery;
        }
        else if (positional == 0 && int.TryParse(a, out int mt))
        {
            maxTokens = mt;
            positional = 1;
        }
        else if (positional <= 1)
        {
            prompt = a;
            positional = 2;
        }
    }

    prompt ??= "Present the player's next choices now. Call the present_choices tool "
               + "with a short prompt and exactly two choices: one story choice and one battle choice.";

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

    if (tokenizer is null)
    {
        Console.WriteLine("OK  meta/config loaded; no tokenizer in file -> tool probe stops here.");
        return 0;
    }

    var format = ToolCallFormatDetector.Resolve(meta, tokenizer);
    Console.WriteLine($"arch={meta.GetString("general.architecture")} format={format} " +
                      $"template={(meta.GetChatTemplate() is null ? "none" : "present")} " +
                      $"temps={string.Join(",", temperatures.Select(t => t.ToString("F2")))} " +
                      $"mode={recovery} reps={reps}");

    try
    {
        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping();
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            LoadMode.Full, quantizedResident: true, maxParallelLoadDegree: 1);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        var agent = new AgentBuilder("Probe").WithTool(
            "present_choices",
            "Present the player's next choices.",
            new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["prompt"] = new JsonObject { ["type"] = "string" },
                    ["choices"] = new JsonObject
                    {
                        ["type"] = "string",
                        ["description"] = "One choice per line: id|label|actionType|detail, for example 1|Enter the cave|story|You step into the dark"
                    }
                },
                ["required"] = new JsonArray("prompt", "choices")
            },
            _ => Task.FromResult("ok"));

        var results = new List<(float Temp, int Passed)>();
        foreach (float temperature in temperatures)
        {
            int passed = 0;
            for (int rep = 1; rep <= reps; rep++)
            {
                Console.WriteLine($"=== temp {temperature:F2} rep {rep}/{reps} ===");
                string? calledTool = null;
                JsonObject? calledArgs = null;
                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(
                    model, tokenizer, meta, agent)
                {
                    MaxTokens = 2048,
                    MaxNewTokens = maxTokens,
                    Temperature = temperature,
                    TopK = 40,
                    TopP = 0.9f,
                    MaxToolCallsPerTurn = 3,
                    ToolCallRecovery = recovery
                };

                session.ProcessToolRequest = (name, toolArgs, _) =>
                {
                    calledTool = name;
                    calledArgs = toolArgs.DeepClone().AsObject();
                    return Task.FromResult(ToolRequestResult.ReturnToCaller());
                };

                session.InitializeChat();

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var text = new System.Text.StringBuilder();
                await foreach (var entry in session.GetResponseStreamAsync(prompt))
                {
                    if (!string.IsNullOrEmpty(entry.Error))
                        Console.WriteLine($"ERR  {entry.Error}");
                    if (entry.Status == ChatStatus.Responding && !string.IsNullOrEmpty(entry.Token))
                        text.Append(entry.Token);
                }
                sw.Stop();

                Console.WriteLine($"--- response ({sw.Elapsed.TotalSeconds:F1}s) ---");
                Console.WriteLine(text.ToString());
                Console.WriteLine("--- raw agent turn ---");
                var lastAgent = session.History.LastOrDefault(message => message.Role == ChatRole.Agent);
                Console.WriteLine(lastAgent?.Content ?? "(none)");
                Console.WriteLine("--- parsed ---");
                Console.WriteLine(calledTool is null
                    ? "tool=none"
                    : $"tool={calledTool} args={calledArgs?.ToJsonString()}");
                if (calledTool is not null)
                    passed++;
            }
            results.Add((temperature, passed));
        }

        Console.WriteLine("--- summary ---");
        foreach (var (temp, passed) in results)
            Console.WriteLine($"temp={temp:F2} reps={reps} passed={passed}");
        int totalPassed = results.Sum(r => r.Passed);
        int totalRuns = results.Count * reps;
        return totalPassed == totalRuns ? 0 : totalPassed == 0 ? 4 : 5;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"toolprobe FAILED: {ex.GetType().Name}: {ex.Message}");
        Console.WriteLine(ex.StackTrace);
        return 1;
    }
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

        var config = GgufLoader.LoadConfig(meta);
        if (config is not null)
        {
            int conv = 0, swa = 0, global = 0;
            for (int i = 0; i < config.NumLayers; i++)
            {
                if (config.IsShortConvLayer(i)) conv++;
                else if (config.IsSwaLayer(i)) swa++;
                else global++;
            }

            Console.WriteLine($"Config: {config.Architecture} layers={config.NumLayers} "
                + $"conv={conv} swa={swa} global={global} window={config.SlidingWindowSize}");
            for (int i = 0; i < config.NumLayers; i++)
            {
                string kind = config.IsShortConvLayer(i) ? "conv"
                    : config.IsSwaLayer(i) ? "swa" : "global";
                Console.WriteLine($"    layer {i:00}: {kind} window={config.WindowSizeForLayer(i)}");
            }
        }

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
    bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
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


        if (b0.SharedGateInp is not null)
        {
            float[] sg = b0.SharedGateInp.Data.ToArray();
            double sum = 0, mx = 0;
            foreach (float v in sg) { sum += Math.Abs(v); if (Math.Abs(v) > mx) mx = Math.Abs(v); }
            Console.WriteLine($"  SharedGateInp avg|v|={sum / sg.Length:F6} max|v|={mx:F6} zeros={sg.Count(v => v == 0f)}/{sg.Length}");
        }
        else
        {
            Console.WriteLine("  SharedGateInp: null (not a shared-expert model)");
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
        Console.WriteLine(ex.StackTrace);
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
    bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
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
        bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
        bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
        bool topk = args.Any(a => a.Equals("topk", StringComparison.OrdinalIgnoreCase));

        // Per-step top-5 logits go to stderr so they interleave with the token
        // stream without polluting OUT:. The std/margin figures are what make this
        // diagnostic rather than just noise: a healthy model gives a peaked
        // distribution (large margin1), while a corrupted one gives a near-uniform
        // soup, and the two are distinguishable at step 0 without a reference.
        GeneratorDiagnostics.DumpTopLogits = topk;
        bool serialMm = args.Any(a => a.Equals("serialmm", StringComparison.OrdinalIgnoreCase));
        var mapping = sharpConfig.ToJigSawMapping(null, serialMm ? false : null);
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
            // llama.cpp's server/CLI default repeat_penalty is 1.0, while this
            // probe's GenerationConfig default is 1.1. "nopenalty" zeroes it so a
            // greedy run is comparable to llama.cpp token-for-token.
            bool noPenalty = args.Any(a => a.Equals("nopenalty", StringComparison.OrdinalIgnoreCase));
            var rawBuilder = new StandardGeneratorBuilder<KVCacherBuilder>();
            var rawGen = rawBuilder.CreateGenerator(model, tokenizer, addBos, addEos: false, caches: null);
            var rawSample = new SamplingConfig { Temperature = 0f, TopK = 1 };
            var rawGenCfg = new GenerationConfig
            {
                MaxNewTokens = maxTokens,
                Stream = true,
                RepetitionPenalty = noPenalty ? 1.0f : new GenerationConfig().RepetitionPenalty,
            };
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

// bench <file> [prompt] [warmup=16] [measure=64] [full] [resident] [serialmm] [nowide] [reps=3] [ids]
//
// Profiler-off decode throughput. Each repetition gets a fresh KV cache and runs
// the same greedy raw generation; the reported window starts after `warmup` and
// covers `measure` tokens, so prefill and first-use JIT are excluded. The median
// across repetitions is the number to compare before/after a kernel change.
static async Task<int> RunBenchAsync(string path, string[] args)
{
    string prompt = args.Length > 2 ? args[2] : "The capital of France is";
    int warmup = args.Length > 3 && int.TryParse(args[3], out int wv) ? wv : 16;
    int measure = args.Length > 4 && int.TryParse(args[4], out int mv) ? mv : 64;
    if (warmup < 1 || measure < 1) { Console.Error.WriteLine("bench needs warmup>=1 and measure>=1"); return 2; }

    if (prompt.Length > 1 && prompt[0] == '@')
    {
        string pf = prompt[1..];
        if (File.Exists(pf)) prompt = File.ReadAllText(pf);
    }

    bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
    bool serialMm = args.Any(a => a.Equals("serialmm", StringComparison.OrdinalIgnoreCase));
    bool noWide = args.Any(a => a.Equals("nowide", StringComparison.OrdinalIgnoreCase));
    bool printIds = args.Any(a => a.Equals("ids", StringComparison.OrdinalIgnoreCase));
    int reps = 3;
    int chunkOverride = 0;
    foreach (string a in args)
    {
        if (a.StartsWith("reps=", StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(a.AsSpan(5), out int r) && r > 0)
            reps = r;
        else if (a.StartsWith("chunks=", StringComparison.OrdinalIgnoreCase) &&
                 int.TryParse(a.AsSpan(7), out int c) && c >= 0)
            chunkOverride = c;
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

    bool oldWide = Q8_0WideWeights.Enabled;
    int oldChunks = Q8_0WideWeights.ParallelChunkOverride;
    try
    {
        if (noWide) Q8_0WideWeights.Enabled = false;
        Q8_0WideWeights.ParallelChunkOverride = chunkOverride;

        var sharpConfig = modelConfig.ForModel();
        var mapping = sharpConfig.ToJigSawMapping(null, serialMm ? false : null);
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident,
            maxParallelLoadDegree: 1);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        var sample = new SamplingConfig { Temperature = 0f, TopK = 1 };
        var results = new List<(double Tps, double Ms, long AllocPerToken, int Steps, string Text, int[] Ids)>(reps);

        for (int rep = 0; rep < reps; rep++)
        {
            using var gen = new StandardGeneratorBuilder<KVCacherBuilder>()
                .CreateGenerator(model, tokenizer, addBos: true, addEos: false, caches: null);
            var genCfg = new GenerationConfig { MaxNewTokens = warmup + measure, Stream = true };

            int consumed = 0;
            long start = 0, allocStart = 0, mainStart = 0;
            int gc0 = 0, gc1 = 0, gc2 = 0;
            var text = new System.Text.StringBuilder();

            await foreach (var frag in gen.GenerateAsync(prompt, sample, genCfg))
            {
                consumed++;
                if (consumed == warmup)
                {
                    start = System.Diagnostics.Stopwatch.GetTimestamp();
                    allocStart = GC.GetTotalAllocatedBytes();
                    gc0 = GC.CollectionCount(0); gc1 = GC.CollectionCount(1); gc2 = GC.CollectionCount(2);
                }
                else if (consumed > warmup)
                {
                    text.Append(frag);
                }

                if (consumed == warmup + measure) break;
            }

            long end = System.Diagnostics.Stopwatch.GetTimestamp();
            if (start == 0)
            {
                Console.WriteLine($"BENCH rep={rep} ended before the {warmup}-token warm-up ({consumed} fragments); skipped");
                continue;
            }

            int steps = Math.Max(0, consumed - warmup);
            if (steps == 0) continue;
            double ms = (end - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            double tps = steps / (ms / 1000.0);
            long allocPerToken = (GC.GetTotalAllocatedBytes() - allocStart) / steps;
            int d0 = GC.CollectionCount(0) - gc0, d1 = GC.CollectionCount(1) - gc1, d2 = GC.CollectionCount(2) - gc2;
            int[] ids = gen.CurrentGeneratedIds is { } all
                ? all.Skip(warmup).Take(steps).ToArray()
                : [];

            results.Add((tps, ms, allocPerToken, steps, text.ToString(), ids));
            Console.WriteLine($"BENCH rep={rep} {tps:F2} tok/s  {ms / steps:F2} ms/tok  {allocPerToken} B/tok  " +
                              $"gc gen0={d0} gen1={d1} gen2={d2}");
        }

        if (results.Count == 0)
        {
            Console.WriteLine("BENCH no measurable decode steps");
            return 4;
        }

        double[] tpsAll = [.. results.Select(r => r.Tps).OrderBy(v => v)];
        double[] msAll = [.. results.Select(r => r.Ms / r.Steps).OrderBy(v => v)];
        long[] allocAll = [.. results.Select(r => r.AllocPerToken).OrderBy(v => v)];
        double medTps = tpsAll[tpsAll.Length / 2];
        double medMs = msAll[msAll.Length / 2];
        long medAlloc = allocAll[allocAll.Length / 2];

        Console.WriteLine($"BENCH median {medTps:F2} tok/s  {medMs:F2} ms/tok  {medAlloc} B/tok  " +
                          $"reps={results.Count} warmup={warmup} measure={measure} " +
                          $"mode={(full ? "Full" : "Streaming")} resident={resident} " +
                          $"serialmm={serialMm} nowide={noWide} " +
                          $"chunks={(chunkOverride == 0 ? "auto" : chunkOverride)} cores={Environment.ProcessorCount}");
        Console.WriteLine($"OUT: {results[0].Text}");
        if (printIds)
            Console.WriteLine($"IDS: {string.Join(',', results[0].Ids)}");
        return 0;
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
    finally
    {
        Q8_0WideWeights.Enabled = oldWide;
        Q8_0WideWeights.ParallelChunkOverride = oldChunks;
    }
}

// benchab <file> [prompt] [warmup=16] [measure=32] [full] [resident]
//
// Full-model A/B under one generator: alternates the wide kernel's pair/single
// choice and 4/8 chunks token by token, so frequency drift hits every variant
// equally. `measure` is tokens per variant, not total.
static async Task<int> RunBenchABAsync(string path, string[] args)
{
    string prompt = args.Length > 2 ? args[2] : "The capital of France is";
    int warmup = args.Length > 3 && int.TryParse(args[3], out int wv) ? wv : 16;
    int measure = args.Length > 4 && int.TryParse(args[4], out int mv) ? mv : 32;
    if (warmup < 1 || measure < 1) { Console.Error.WriteLine("benchab needs warmup>=1 and measure>=1"); return 2; }

    bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
    bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));

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
        Console.WriteLine("OK  meta/config loaded; no tokenizer in file -> verification stops here.");
        return 0;
    }

    var variants = new (string Name, bool Pair, int Chunks)[]
    {
        ("pair/auto", true, 0), ("single/auto", false, 0),
        ("pair/8", true, 8), ("single/8", false, 8),
    };
    var times = new List<double>[variants.Length];
    for (int i = 0; i < times.Length; i++) times[i] = new List<double>(measure);

    bool oldPair = Q8_0WideWeights.UsePairKernel;
    int oldChunks = Q8_0WideWeights.ParallelChunkOverride;
    try
    {
        var sharpConfig = modelConfig.ForModel();
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(sharpConfig.ToJigSawMapping(null, null)), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident,
            maxParallelLoadDegree: 1);
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, sharpConfig.ToJigSawMapping(null, null));

        using var gen = new StandardGeneratorBuilder<KVCacherBuilder>()
            .CreateGenerator(model, tokenizer, addBos: true, addEos: false, caches: null);
        var sample = new SamplingConfig { Temperature = 0f, TopK = 1 };
        int total = warmup + measure * variants.Length;
        var genCfg = new GenerationConfig { MaxNewTokens = total, Stream = true };

        int consumed = 0, measured = 0, current = 0;
        long prev = 0;
        Q8_0WideWeights.UsePairKernel = variants[0].Pair;
        Q8_0WideWeights.ParallelChunkOverride = variants[0].Chunks;

        await foreach (var _ in gen.GenerateAsync(prompt, sample, genCfg))
        {
            consumed++;
            long now = System.Diagnostics.Stopwatch.GetTimestamp();

            if (consumed == warmup)
            {
                prev = now;
                measured = 0;
                current = 0;
                Q8_0WideWeights.UsePairKernel = variants[0].Pair;
                Q8_0WideWeights.ParallelChunkOverride = variants[0].Chunks;
            }
            else if (consumed > warmup)
            {
                double ms = (now - prev) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                times[current].Add(ms);
                prev = now;
                measured++;

                current = measured % variants.Length;
                Q8_0WideWeights.UsePairKernel = variants[current].Pair;
                Q8_0WideWeights.ParallelChunkOverride = variants[current].Chunks;
            }

            if (consumed == total) break;
        }

        Console.WriteLine($"AB {Path.GetFileName(path)} prompt=\"{prompt}\" warmup={warmup} " +
                          $"mode={(full ? "Full" : "Streaming")} resident={resident} cores={Environment.ProcessorCount}");
        for (int v = 0; v < variants.Length; v++)
        {
            var list = times[v];
            if (list.Count == 0) { Console.WriteLine($"  {variants[v].Name,-12} no samples"); continue; }
            double[] sorted = [.. list.OrderBy(x => x)];
            double median = sorted[sorted.Length / 2];
            double mean = list.Average();
            Console.WriteLine($"  {variants[v].Name,-12} n={list.Count,3} median={median:F2} ms/tok  " +
                              $"mean={mean:F2}  {1000.0 / median:F2} tok/s");
        }
        return 0;
    }
    catch (NotSupportedException ex)
    {
        Console.WriteLine($"\nUNSUPPORTED (loud fail): {ex.Message}");
        return 3;
    }
    finally
    {
        Q8_0WideWeights.UsePairKernel = oldPair;
        Q8_0WideWeights.ParallelChunkOverride = oldChunks;
    }
}

// stages <file> [prompt] [warmup=16] [measure=48] [full] [resident] [serialmm]
//
// Per-stage breakdown of CPU decode, powered by DecodeProfiler (opt-in marks
// throughout SharpMind.Model and StandardGenerator; a profiled run is slower
// than a real one). Prefill and JIT promotion land in a warm-up window with the
// profiler off; the stage table below covers only single-token decode steps.
//
// Flow: one greedy raw GenerateAsync over warmup+measure tokens. The profiler is
// flipped on while the generator is suspended between yields, so the measured
// window is exactly the decode steps that start after the warm-up fragment — no
// prefill, no prompt re-prefix, no second generator pass. GC collections and
// allocated bytes are sampled over the same window and reported per token.
static async Task<int> RunStagesAsync(string path, string[] args)
{
    string prompt = args.Length > 2 ? args[2] : "Hello!";
    int warmup = args.Length > 3 && int.TryParse(args[3], out int wv) ? wv : 16;
    int measure = args.Length > 4 && int.TryParse(args[4], out int mv) ? mv : 48;
    if (warmup < 1 || measure < 1) { Console.Error.WriteLine("stages needs warmup>=1 and measure>=1"); return 2; }

    // "@path" reads the prompt from a file, matching `run`.
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
        bool full = !args.Any(a => a.Equals("streaming", StringComparison.OrdinalIgnoreCase));
        bool resident = args.Any(a => a.Equals("resident", StringComparison.OrdinalIgnoreCase));
        bool serialMm = args.Any(a => a.Equals("serialmm", StringComparison.OrdinalIgnoreCase));
        int window = -1;
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i].Equals("window", StringComparison.OrdinalIgnoreCase) && int.TryParse(args[i + 1], out int w))
                window = w;
        var mapping = sharpConfig.ToJigSawMapping(null, serialMm ? false : null);
        using var weights = ModelFactory.CreateWeights(
            modelConfig, sharpConfig, QuantizationFactory.Create(mapping), path,
            full ? LoadMode.Full : LoadMode.Streaming, quantizedResident: resident);
        if (!full && window > 0 && weights is TransformerWeightsStreaming sw)
            sw.ResidentWindow = window;
        weights.InitializeWeights();
        using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);

        // Greedy raw generation, no chat template — same path the run mode uses for "raw".
        var rawBuilder = new StandardGeneratorBuilder<KVCacherBuilder>();
        var rawGen = rawBuilder.CreateGenerator(model, tokenizer, addBos: true, addEos: false, caches: null);
        var sample = new SamplingConfig { Temperature = 0f, TopK = 1 };
        var genCfg = new GenerationConfig { MaxNewTokens = warmup + measure, Stream = true };

        DecodeProfiler.Reset();
        DecodeProfiler.Enabled = false;

        int consumed = 0;
        var text = new System.Text.StringBuilder();
        long wallStart = 0, wallEnd = 0;
        long gc0 = 0, gc1 = 0, gc2 = 0, allocStart = 0, mainStart = 0;
        long _liveStart = 0, _privStart = 0;
        bool windowOpened = false;
        long prevStamp = 0, prevAlloc = 0;
        var perStep = new List<(double Ms, double MiB)>(measure);

        await foreach (var frag in rawGen.GenerateAsync(prompt, sample, genCfg))
        {
            consumed++;
            if (consumed == warmup)
            {
                windowOpened = true;
                DecodeProfiler.Reset();
                DecodeProfiler.Enabled = true;
                wallStart = System.Diagnostics.Stopwatch.GetTimestamp();
                gc0 = GC.CollectionCount(0); gc1 = GC.CollectionCount(1); gc2 = GC.CollectionCount(2);
                allocStart = GC.GetTotalAllocatedBytes();
                mainStart = GC.GetAllocatedBytesForCurrentThread();
                _liveStart = GC.GetTotalMemory(forceFullCollection: false);
                _privStart = System.Diagnostics.Process.GetCurrentProcess().PrivateMemorySize64;
                prevStamp = wallStart;
                prevAlloc = allocStart;
            }
            else if (consumed > warmup)
            {
                long now = System.Diagnostics.Stopwatch.GetTimestamp();
                long nowAlloc = GC.GetTotalAllocatedBytes();
                perStep.Add(((now - prevStamp) * 1000.0 / System.Diagnostics.Stopwatch.Frequency,
                             (nowAlloc - prevAlloc) / (1024.0 * 1024.0)));
                prevStamp = now;
                prevAlloc = nowAlloc;
            }
            if (consumed > warmup) text.Append(frag);
            if (consumed == warmup + measure)
            {
                DecodeProfiler.Enabled = false;
                wallEnd = System.Diagnostics.Stopwatch.GetTimestamp();
                break;
            }
        }

        if (!windowOpened)
        {
            Console.WriteLine($"generation ended before the {warmup}-token warm-up ({consumed} fragments); nothing to measure.");
            return 4;
        }
        if (wallEnd == 0) wallEnd = System.Diagnostics.Stopwatch.GetTimestamp();
        DecodeProfiler.Enabled = false;

        double wallMs = (wallEnd - wallStart) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        long steps = DecodeProfiler.Count(DecodeStage.Forward);
        double tps = wallMs > 0 ? steps / (wallMs / 1000.0) : 0;
        long d0 = GC.CollectionCount(0) - gc0, d1 = GC.CollectionCount(1) - gc1, d2 = GC.CollectionCount(2) - gc2;
        double allocMiB = (GC.GetTotalAllocatedBytes() - allocStart) / (1024.0 * 1024.0);
        double allocPerTok = steps > 0 ? (GC.GetTotalAllocatedBytes() - allocStart) / (double)steps : 0;

        Console.WriteLine($"OUT: {text}");
        Console.WriteLine($"--- decode stages: {Path.GetFileName(path)}  prompt=\"{prompt}\"  warmup={warmup}  measure={measure}  greedy ---");
        // Footprint is the retained state (managed live bytes + committed private memory) at
        // steady state, NOT the alloc churn printed below. In streaming the pool caps at
        // (ResidentWindow+2) x the largest layer, so live memory should stay far below full
        // mode's whole-model weight arrays no matter how many MiB/token it churns. Compare
        // the start/end deltas: a leak would show as growth across the measured window.
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        long liveEnd = GC.GetTotalMemory(forceFullCollection: false);
        long privEnd = proc.PrivateMemorySize64;
        long liveStart = 0, privStart = 0;
        if (allocStart > 0) { liveStart = _liveStart; privStart = _privStart; }
        Console.WriteLine($"footprint  start live={liveStart / (1024.0 * 1024.0):F0} MiB priv={privStart / (1024.0 * 1024.0):F0} MiB  -> end live={liveEnd / (1024.0 * 1024.0):F0} MiB priv={privEnd / (1024.0 * 1024.0):F0} MiB  (delta {(liveEnd - liveStart) / 1024.0 / 1024.0:F0} MiB live / window)");

        Console.WriteLine($"wall {wallMs / 1000.0:F2}s  {steps} decode steps  {tps:F2} tok/s  |  gc gen0={d0} gen1={d1} gen2={d2}  |  alloc {allocMiB:F1} MiB ({allocPerTok:F0} B/tok)");
        Console.WriteLine($"main-thread alloc {(GC.GetAllocatedBytesForCurrentThread() - mainStart) / (1024.0 * 1024.0):F1} MiB");
        if (perStep.Count > 0)
        {
            double msAvg = perStep.Average(p => p.Ms), msMin = perStep.Min(p => p.Ms), msMax = perStep.Max(p => p.Ms);
            double mbAvg = perStep.Average(p => p.MiB), mbMin = perStep.Min(p => p.MiB), mbMax = perStep.Max(p => p.MiB);
            Console.WriteLine($"per-step  ms min/avg/max {msMin:F1}/{msAvg:F1}/{msMax:F1}   alloc MiB min/avg/max {mbMin:F1}/{mbAvg:F1}/{mbMax:F1}");
        }
        Console.WriteLine(DecodeProfiler.Format("decode", wallMs));
        if (weights is TransformerWeightsStreaming streamWeights)
            Console.WriteLine($"pool  retained={streamWeights.PoolRetainedBytes / (1024.0 * 1024.0):F1} MiB  hits={streamWeights.PoolHits}  misses={streamWeights.PoolMisses}");
        Console.WriteLine($"note: profiled tok/s is slower than real; measure throughput with `run raw` and the profiler off.");
        return steps > 0 ? 0 : 4;
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

// widebench [K=896] [N=4864] [iters=200]
//
// Exercises the production Q8_0 wide decode path (scratch + chunk scheduling)
// without a model. Short kbench-style runs are dominated by timer noise, so this
// warms up past tiered-JIT promotion and reports a long-run MAC rate.
static unsafe int RunWideKernelBench(string[] args)
{
    int K = args.Length > 1 && int.TryParse(args[1], out int kv) ? kv : 896;
    int N = args.Length > 2 && int.TryParse(args[2], out int nv) ? nv : 4864;
    int iters = args.Length > 3 && int.TryParse(args[3], out int iv) ? iv : 200;
    int chunks = args.Length > 4 && int.TryParse(args[4], out int cv) ? cv : 0;
    if (K <= 0 || N <= 0 || iters <= 0 || K % 32 != 0 || chunks < 0)
    {
        Console.Error.WriteLine("widebench needs K>0 (multiple of 32), N>0, iters>0, chunks>=0");
        return 2;
    }

    if (!Q8_0WideWeights.IsSupported(K, N))
    {
        Console.WriteLine($"WIDE unsupported K={K} N={N}");
        return 3;
    }

    int rawBlockBytes = 34;
    long wBytes = (long)N * (K / 32) * rawBlockBytes;
    if (wBytes > Array.MaxLength) { Console.Error.WriteLine("widebench weights too large"); return 2; }

    byte[] raw = new byte[wBytes];
    for (long i = 0; i < wBytes; i++)
        raw[i] = (i & 1) == 0 ? (byte)0x11 : (byte)0x22;

    Q8_0WideWeights.ParallelChunkOverride = chunks;
    var wide = Q8_0WideWeights.TryCreate(raw, K, N);
    if (wide is null) { Console.WriteLine($"WIDE could not build K={K} N={N}"); return 3; }

    float* input = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
        (nuint)((long)K * sizeof(float)), 64);
    float* output = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(
        (nuint)((long)N * sizeof(float)), 64);
    try
    {
        for (int i = 0; i < K; i++) input[i] = 1f;
        for (int i = 0; i < N; i++) output[i] = 0f;

        for (int i = 0; i < 60; i++) wide.MatMul(input, output, 1);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) wide.MatMul(input, output, 1);
        sw.Stop();

        double seconds = sw.Elapsed.TotalSeconds;
        double macs = (double)K * N * iters;
        Console.WriteLine($"WIDE K={K} N={N} iters={iters} chunks={(chunks == 0 ? "auto" : chunks)} " +
                          $"cores={Environment.ProcessorCount} {macs / seconds / 1e9:F3} GMAC/s");
        return 0;
    }
    finally
    {
        System.Runtime.InteropServices.NativeMemory.AlignedFree(input);
        System.Runtime.InteropServices.NativeMemory.AlignedFree(output);
    }
}

// kbench <kind> [K=2048] [N=1408] [iters=50] [fma|avx2|scalar]
//
// Runs one quantized decode matmul (M=1) in isolation on a deliberately
// non-zero weight buffer and reports the achieved MAC rate. An all-zero buffer
// would let the FMA take a zero-input path and make the kernel look faster than
// it is on real weights, so the bytes alternate 0x11/0x22 — half fields that are
// never 0 or 31, hence never a denormal or a NaN. This separates "the kernel is
// slow" from "the orchestration around it is slow", which a whole forward pass
// cannot do because the two are multiplied together.
static unsafe int RunKernelBench(string[] args)
{
    string kind = args.Length > 1 ? args[1].ToLowerInvariant() : "q2k";
    int K = args.Length > 2 && int.TryParse(args[2], out int kv) ? kv : 2048;
    int N = args.Length > 3 && int.TryParse(args[3], out int nv) ? nv : 1408;
    int iters = args.Length > 4 && int.TryParse(args[4], out int iv) ? iv : 50;
    string variant = args.Length > 5 ? args[5].ToLowerInvariant() : "fma";

    int elPerBlock; int blockBytes;
    switch (kind)
    {
        case "q2k": elPerBlock = 256; blockBytes = 84; break;
        case "q3k": elPerBlock = 256; blockBytes = 110; break;
        case "q4k": elPerBlock = 256; blockBytes = 144; break;
        case "q5k": elPerBlock = 256; blockBytes = 176; break;
        case "q6k": elPerBlock = 256; blockBytes = 210; break;
        case "q8k": elPerBlock = 256; blockBytes = 292; break;
        case "q4_0": elPerBlock = 32; blockBytes = 18; break;
        case "q4_1": elPerBlock = 32; blockBytes = 20; break;
        case "iq4nl": elPerBlock = 32; blockBytes = 18; break;
        case "q5_0": elPerBlock = 32; blockBytes = 22; break;
        case "q5_1": elPerBlock = 32; blockBytes = 24; break;
        case "q8_0": elPerBlock = 32; blockBytes = 34; break;
        case "q8_1": elPerBlock = 32; blockBytes = 36; break;
        case "f32": elPerBlock = 1; blockBytes = 4; break;
        default:
            Console.Error.WriteLine($"kbench: unknown kind '{kind}'");
            return 2;
    }

    string key = variant == "fma" ? kind : $"{kind}_{variant}";
    QuantizedMatMulFn par; QuantizedMatMulFn ser;
    switch (key)
    {
        case "q2k": par = QuantizationKernels.QuantizedMatMulQ2K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ2K_Serial_FMA; break;
        case "q2k_avx2": par = QuantizationKernels.QuantizedMatMulQ2K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ2K_Serial_AVX2; break;
        case "q2k_scalar": par = QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar; break;
        case "q3k": par = QuantizationKernels.QuantizedMatMulQ3K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ3K_Serial_FMA; break;
        case "q3k_avx2": par = QuantizationKernels.QuantizedMatMulQ3K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ3K_Serial_AVX2; break;
        case "q3k_scalar": par = QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar; break;
        case "q6k": par = QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA; break;
        case "q6k_avx2": par = QuantizationKernels.QuantizedMatMulQ6K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ6K_Serial_AVX2; break;
        case "q6k_scalar": par = QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar; break;
        case "iq4nl": par = QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ4_NL_Serial_FMA; break;
        case "iq4nl_avx2": par = QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2; break;
        case "iq4nl_scalar": par = QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar; break;
        case "q4k": par = QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA; break;
        case "q4k_avx2": par = QuantizationKernels.QuantizedMatMulQ4K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ4K_Serial_AVX2; break;
        case "q4k_scalar": par = QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar; break;
        case "q5k": par = QuantizationKernels.QuantizedMatMulQ5K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ5K_Serial_FMA; break;
        case "q5k_avx2": par = QuantizationKernels.QuantizedMatMulQ5K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ5K_Serial_AVX2; break;
        case "q5k_scalar": par = QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar; break;
        case "q4_0": par = QuantizationKernels.QuantizedMatMulQ4_0_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ4_0_Serial_FMA; break;
        case "q4_0_avx2": par = QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2; break;
        case "q4_0_scalar": par = QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar; break;
        case "q4_1": par = QuantizationKernels.QuantizedMatMulQ4_1_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ4_1_Serial_FMA; break;
        case "q4_1_avx2": par = QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2; break;
        case "q4_1_scalar": par = QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar; break;
        case "q5_0": par = QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA; break;
        case "q5_0_avx2": par = QuantizationKernels.QuantizedMatMulQ5_0_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ5_0_Serial_AVX2; break;
        case "q5_0_scalar": par = QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar; break;
        case "q5_1": par = QuantizationKernels.QuantizedMatMulQ5_1_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ5_1_Serial_FMA; break;
        case "q5_1_avx2": par = QuantizationKernels.QuantizedMatMulQ5_1_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ5_1_Serial_AVX2; break;
        case "q5_1_scalar": par = QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar; break;
        case "q8_1": par = QuantizationKernels.QuantizedMatMulQ8_1_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ8_1_Serial_FMA; break;
        case "q8_1_avx2": par = QuantizationKernels.QuantizedMatMulQ8_1_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ8_1_Serial_AVX2; break;
        case "q8_1_scalar": par = QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar; break;
        case "q8k": par = QuantizationKernels.QuantizedMatMulQ8K_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ8K_Serial_FMA; break;
        case "q8k_avx2": par = QuantizationKernels.QuantizedMatMulQ8K_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ8K_Serial_AVX2; break;
        case "q8k_scalar": par = QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar; break;
        case "q8_0": par = QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA; break;
        case "q8_0_avx2": par = QuantizationKernels.QuantizedMatMulQ8_0_Parallel_AVX2; ser = QuantizationKernels.QuantizedMatMulQ8_0_Serial_AVX2; break;
        case "q8_0_scalar": par = QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar; break;
        case "f32": par = QuantizationKernels.QuantizedMatMulF32_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulF32_Serial_FMA; break;
        case "f32_avx2": par = QuantizationKernels.QuantizedMatMulF32_Parallel_FMA; ser = QuantizationKernels.QuantizedMatMulF32_Serial_FMA; break;
        case "f32_scalar": par = QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar; ser = QuantizationKernels.QuantizedMatMulF32_Serial_Scalar; break;
        default:
            Console.Error.WriteLine($"kbench: unknown variant '{key}'");
            return 2;
    }

    long wBytes = ((long)N * K / elPerBlock) * blockBytes;
    byte* raw = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)wBytes, 64);
    float* input = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)((long)K * sizeof(float)), 64);
    float* output = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)((long)N * sizeof(float)), 64);
    try
    {
        // Non-zero weights with halves whose exponent field is never 0 or 31:
        // an all-zero buffer lets the FMA take a zero-input path, which makes the
        // kernel look faster than it is on real weights.
        for (long i = 0; i < wBytes; i++) raw[i] = (i & 1) == 0 ? (byte)0x11 : (byte)0x22;
        for (int i = 0; i < K; i++) input[i] = 1f;

        double macs = (double)K * N;
        Console.WriteLine($"KIND={key} K={K} N={N} wbytes={wBytes} iters={iters} " +
                          $"cores={Environment.ProcessorCount}");

        par(input, raw, output, 1, K, N);
        ser(input, raw, output, 1, K, N);
        // Tiered JIT promotes a method after ~30 invocations; warming up well past
        // that keeps the promotion out of the timed window (it otherwise lands
        // mid-run and makes the two measurements incomparable).
        for (int i = 0; i < 60; i++) { par(input, raw, output, 1, K, N); ser(input, raw, output, 1, K, N); }
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) par(input, raw, output, 1, K, N);
        sw.Stop();
        double tPar = sw.Elapsed.TotalSeconds;

        sw.Restart();
        for (int i = 0; i < iters; i++) ser(input, raw, output, 1, K, N);
        sw.Stop();
        double tSer = sw.Elapsed.TotalSeconds;

        Console.WriteLine($"  par {tPar:F4}s {macs * iters / tPar / 1e9:F3} GMAC/s");
        Console.WriteLine($"  ser {tSer:F4}s {macs * iters / tSer / 1e9:F3} GMAC/s");
        return 0;
    }
    finally
    {
        System.Runtime.InteropServices.NativeMemory.AlignedFree(raw);
        System.Runtime.InteropServices.NativeMemory.AlignedFree(input);
        System.Runtime.InteropServices.NativeMemory.AlignedFree(output);
    }
}