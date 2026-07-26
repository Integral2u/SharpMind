using SharpMind.Core.Quantization;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SandBox.RunModels
{
    public class DiagnosticModelRunner
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt, string[] Models, bool withGPU = false)
        {
            var totalTime = Stopwatch.StartNew();
            foreach (var m in Models)
            {
                Console.ForegroundColor = ConsoleColor.White;
                var returnedPrompt = false;
                var tok = 0;
                CancellationTokenSource cancellationTokenSource = new();
                string modelPath = string.Empty;
                ModelFormat? fmt = null;
                foreach (var mFmt in Enum.GetValues<ModelFormat>())
                {
                    var ext = ModelFormatHelpers.GetExtension(mFmt);
                    modelPath = Path.Combine(ModelPath, $"{m}{ext}");
                    if (File.Exists(modelPath))
                    {
                        fmt = mFmt; break;
                    }
                }
                if (fmt == null) continue;
                var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);

                await Console.Out.WriteLineAsync($"Testing {m}");

                await Console.Out.FlushAsync();
                metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                await Console.Out.WriteLineAsync($"[DIAG] {meta}: arch='{meta.GetString("general.architecture")}'");
                await Console.Out.WriteLineAsync($"[DIAG] config: hidden={modelConfig.HiddenDim} layers={modelConfig.NumLayers} heads={modelConfig.NumHeads} kv={modelConfig.NumKvHeads} ffn={modelConfig.FfnDim} headDim={modelConfig.HeadDim} headDimOverride={modelConfig.HeadDimOverride} maxSeq={modelConfig.MaxSeqLen} ropeTheta={modelConfig.RopeTheta} ropeDim={modelConfig.RopeDim} tieEmb={modelConfig.TieWordEmbeddings}");
                // Dump metadata KV pairs matching rope or head_dim
                if (meta?.KvPairs != null)
                {
                    var kv = meta.KvPairs.Where(k => k.Key.Contains("rope", StringComparison.OrdinalIgnoreCase) || k.Key.Contains("head_dim", StringComparison.OrdinalIgnoreCase) || k.Key.Contains("dimension", StringComparison.OrdinalIgnoreCase));
                    foreach (var k in kv)
                        await Console.Out.WriteLineAsync($"[DIAG] meta {k.Key} = {k.Value}");
                }
                // Dump first 30 tensor names
                if (meta?.Tensors != null)
                {
                    var firstTensors = meta.Tensors.Take(30).Select(t => $"{t.Name}[{string.Join("x", t.Shape)}]");
                    await Console.Out.WriteLineAsync($"[DIAG] tensors: {string.Join(", ", firstTensors)}");
                }
                // Diagnostic: dump rendered prompt and formatter errors
                var formatter = ChatPromptFormatterFactory.Create(meta);
                if (formatter is JinjaTemplateFormatter jinja)
                {
                    var testMsgs = new[] { ChatMessage.User(prompt) };
                    string rendered = jinja.Format(testMsgs, tokenizer, false);
                    Console.Error.WriteLine($"[formatter] prompt={rendered.Replace("\n", "\\n")}");
                    if (jinja.LastErrors is { Count: > 0 })
                        foreach (var err in jinja.LastErrors)
                            Console.Error.WriteLine($"[formatter ERROR] {err}");
                }
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel();
                // Build a single combined mapping. WithGpu() now overrides quant ops
                // as well as model-level ops — no separate qOpsMapping needed.
                var mapping = withGPU ? new SharpMind.MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .WithGpu()
                    .Build() :
                    new SharpMind.MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .Build();

                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var qOps = QuantizationFactory.Create(mapping);
                using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath);
                weights.InitializeWeights();

                await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                // Dump weight info for blocks 0,1,2 (Gemma-3)
                for (int bi = 0; bi < Math.Min(3, weights.Blocks.Length); bi++)
                {
                    var b = weights.Blocks[bi];
                    var n1 = b.Norm1W; var n2 = b.Norm2W;
                    var p1 = b.PostNorm1W; var p2 = b.PostNorm2W;
                    var qn = b.QNormW; var kn = b.KNormW;
                    if (n1 != null) { var d = n1.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.Norm1W: first5={d[0]:G4},{d[1]:G4},{d[2]:G4},{d[3]:G4},{d[4]:G4} last={d[n1.ElementCount-1]:G4}"); }
                    if (n2 != null) { var d = n2.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.Norm2W: first5={d[0]:G4},{d[1]:G4},{d[2]:G4},{d[3]:G4},{d[4]:G4} last={d[n2.ElementCount-1]:G4}"); }
                    if (p1 != null) { var d = p1.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.PostNorm1W: first5={d[0]:G4},{d[1]:G4},{d[2]:G4},{d[3]:G4},{d[4]:G4} last={d[p1.ElementCount-1]:G4}"); }
                    if (p2 != null) { var d = p2.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.PostNorm2W: first5={d[0]:G4},{d[1]:G4},{d[2]:G4},{d[3]:G4},{d[4]:G4} last={d[p2.ElementCount-1]:G4}"); }
                    if (qn != null) { var d = qn.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.QNormW: first3={d[0]:G4},{d[1]:G4},{d[2]:G4} last={d[qn.ElementCount-1]:G4}"); }
                    if (kn != null) { var d = kn.Data; await Console.Out.WriteLineAsync($"[DIAG] blk{bi}.KNormW: first3={d[0]:G4},{d[1]:G4},{d[2]:G4} last={d[kn.ElementCount-1]:G4}"); }
                }
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateTransformer executed in: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Restart();

                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                {
                    MaxTokens = 256,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                await Console.Out.WriteLineAsync($"ChatSession executed in: {sw.Elapsed.TotalSeconds:F2}s");
                sw.Stop();
                var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);


                async void Response(ChatStreamEntry text)
                {
                    Console.ForegroundColor = text.Status == ChatStatus.Thinking ? ConsoleColor.Gray : ConsoleColor.Blue;
                    await Console.Out.WriteAsync(text.Token);
                    tok++;
                    //if (tok > 60) cancellationTokenSource.Cancel();
                }
                async Task<ChatMessage> Prompt()
                {
                    if (!returnedPrompt && !cancellationTokenSource.IsCancellationRequested)
                    {
                        returnedPrompt = true;
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        await Console.Out.WriteLineAsync($"Prompt:{prompt}");
                        await Console.Out.FlushAsync();
                        await Console.Out.WriteAsync("Response:");
                        return new ChatMessage { Content = prompt, Role = ChatRole.User };
                    }
                    await Console.Out.WriteLineAsync();
                    await Console.Out.FlushAsync();
                    cancellationTokenSource.Cancel();
                    return new ChatMessage { Content = "exit", Role = ChatRole.User };
                }
                await Console.Out.WriteLineAsync();
                await Console.Out.WriteLineAsync($"Tokens per second: {session.TokensPerSecond ?? 0:F2}  TTFT: {session.TimeToFirstToken?.ToString("F3") ?? "N/A"}s");
            }
            await Console.Out.WriteLineAsync($"All Models Executed in: {totalTime.Elapsed.TotalSeconds:F2}s");
            await Console.Out.WriteLineAsync();
        }
    }
}
