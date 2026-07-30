using SharpMind.Core;
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
using System.Text;
using System.Threading;

namespace SandBox.RunModels
{
    public class TokenDiagnosticRunner
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

        public static async Task RunAsync(string userPrompt, string[] models, int maxGenTokens = 50)
        {
            foreach (var m in models)
            {
                Console.ForegroundColor = ConsoleColor.White;
                string modelPath = string.Empty;
                ModelFormat? fmt = null;
                foreach (var mFmt in Enum.GetValues<ModelFormat>())
                {
                    var ext = ModelFormatHelpers.GetExtension(mFmt);
                    modelPath = Path.Combine(ModelPath, $"{m}{ext}");
                    if (File.Exists(modelPath)) { fmt = mFmt; break; }
                }
                if (fmt == null) { Console.Error.WriteLine($"Model '{m}' not found"); continue; }

                var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
                await Console.Out.WriteLineAsync($"\n========== {m} ==========");
                metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null) { Console.Error.WriteLine("No tokenizer"); continue; }

                // --- 1. Dump chat template ---
                var template = meta.GetChatTemplate();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                await Console.Out.WriteLineAsync($"Template: {template?.Replace("\n", "\\n")}");
                Console.ForegroundColor = ConsoleColor.White;

                // --- 2. Render prompt using formatter ---
                var formatter = ChatPromptFormatterFactory.Create(meta);
                string rendered = userPrompt;
                if (formatter is JinjaTemplateFormatter jinja)
                {
                    var msgs = new[] { ChatMessage.User(userPrompt) };
                    rendered = jinja.Format(msgs, tokenizer, addBos: true);
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    await Console.Out.WriteLineAsync($"Rendered prompt ({rendered.Length} chars):");
                    await Console.Out.WriteLineAsync($"  escaped: {rendered.Replace("\n", "\\n").Replace("\r", "\\r")}");
                    if (jinja.LastErrors is { Count: > 0 })
                    {
                        Console.ForegroundColor = ConsoleColor.Red;
                        foreach (var e in jinja.LastErrors)
                            await Console.Out.WriteLineAsync($"  [formatter error] {e}");
                    }
                    Console.ForegroundColor = ConsoleColor.White;
                }

                // --- 3. Dump token IDs of rendered prompt ---
                var promptIds = tokenizer.Encode(rendered, addBos: false, addEos: false);
                await Console.Out.WriteLineAsync($"Prompt token IDs ({promptIds.Length} total, first 30):");
                int[] singleBuf = new int[1];
                for (int i = 0; i < Math.Min(30, promptIds.Length); i++)
                {
                    int id = promptIds[i];
                    singleBuf[0] = id;
                    string rawDecoded = tokenizer.Decode(singleBuf.AsSpan(), skipSpecials: false);
                    string cleanDecoded = tokenizer.Decode(singleBuf.AsSpan(), skipSpecials: true);
                    await Console.Out.WriteLineAsync($"  [{i,3}] {id,6} -> raw:'{rawDecoded.Replace("\n", "\\n").Replace("\r", "\\r")}' clean:'{cleanDecoded.Replace("\n", "\\n").Replace("\r", "\\r")}'");
                }

                // --- 4. Dump stop token IDs ---
                var stopIds = tokenizer.GetEndOfGenerationIds();
                await Console.Out.WriteLineAsync($"Stop token IDs ({stopIds.Count}):");
                foreach (int sid in stopIds)
                {
                    string t = tokenizer.IdToToken(sid).Replace("\n", "\\n");
                    await Console.Out.WriteLineAsync($"  {sid,6} : '{t}'");
                }

                // --- 5. Build model ---
                var sharpConfig = modelConfig.ForModel();
                var mapping = new MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .Build();

                GC.Collect(); GC.WaitForPendingFinalizers();

                var sw = Stopwatch.StartNew();
                var qOps = QuantizationFactory.Create(mapping);
                using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath);
                weights.InitializeWeights();
                await Console.Out.WriteLineAsync($"Load+Init: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Restart();
                using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
                var addBos = ModelMetaData.ResolveAddBos(meta, tokenizer.UseSentencePieceMerge);
                var addEos = ModelMetaData.ResolveAddEos(meta);

                var genCfg = GenerationConfig.Chat(tokenizer.EosId) with
                {
                    MaxNewTokens = maxGenTokens,
                    RepetitionPenalty = 1.0f,
                    StopTokenIds = stopIds,
                };
                var sampleCfg = SamplingConfig.Greedy;

                using var generator = new StandardGenerator<KVCacherBuilder>(model, tokenizer, addBos, addEos);
                await Console.Out.WriteLineAsync($"Model+Generator: {sw.Elapsed.TotalSeconds:F2}s");

                // --- 6. Generate and log every token ---
                sw.Restart();
                var decodedSb = new StringBuilder();
                var tokenLog = new List<(int Id, string Raw, string Clean)>();
                int genCount = 0;

                generator.OnTokenGenerated = (id) =>
                {
                    if (genCount >= maxGenTokens) return;
                    singleBuf[0] = id;
                    string rawFrag = tokenizer.Decode(singleBuf.AsSpan(), skipSpecials: false);
                    string cleanFrag = tokenizer.Decode(singleBuf.AsSpan(), skipSpecials: true);
                    lock (tokenLog) tokenLog.Add((id, rawFrag, cleanFrag));
                    Interlocked.Increment(ref genCount);
                };

                await foreach (var fragment in generator.GenerateFromTokensAsync(promptIds, sampleCfg, genCfg))
                {
                    decodedSb.Append(fragment);
                }

                // --- 7. Print token log ---
                await Console.Out.WriteLineAsync($"\nGenerated tokens ({tokenLog.Count}):");
                for (int i = 0; i < tokenLog.Count; i++)
                {
                    var (id, raw, clean) = tokenLog[i];
                    bool isStop = stopIds.Contains(id);
                    string marker = isStop ? " [STOP]" : "";
                    await Console.Out.WriteLineAsync(
                        $"  [{i,3}] {id,6} -> " +
                        $"raw:'{raw.Replace("\n", "\\n").Replace("\r", "\\r")}' " +
                        $"clean:'{clean.Replace("\n", "\\n").Replace("\r", "\\r")}'{marker}");
                }

                // --- 8. Show final decoded output ---
                Console.ForegroundColor = ConsoleColor.Green;
                await Console.Out.WriteLineAsync($"\nFinal decoded output ({decodedSb.Length} chars):");
                await Console.Out.WriteLineAsync($"'{decodedSb.ToString().Replace("\n", "\\n").Replace("\r", "\\r")}'");
                Console.ForegroundColor = ConsoleColor.White;
                sw.Stop();
                await Console.Out.WriteLineAsync($"Generation: {sw.Elapsed.TotalSeconds:F2}s, {tokenLog.Count} tokens");
            }
        }
    }
}
