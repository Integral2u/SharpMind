using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;


namespace SharpMind.Samples.Examples
{
    public class KnownFailingModels
    {
        private static readonly string[] Models =
        [            
            "Qwen3-0.6B-Q8_0",      //Response:.json
            "Qwen3-0.6B-Q6_K",      //Lots of new lines \n Response: 
            "Qwen3-0.6B-Q5_K_M",    //Response:zym???yms Trustees stands????iclesYT???.dup? searchData
            "Qwen3-0.6B-Q4_K_M",    //Lots of new lines \n Response: 
            "Qwen3-0.6B-Q4_1",      //Response: {\nohic?? Gst","#toophonieres?? verst???uably (oise
            "Qwen3-0.6B-Q4_0",      //Response:);}\n\nYSzo(fin industri);}\n\n Cain?enetuserService vig.getLabel.LAitorezburg
            "Qwen3-0.6B-Q3_K_M",    //Response:oon uiisión adjacent»?t harmful applicable  of isClass} robber present
            "Qwen3-0.6B-Q2_K",      //Response:[floattimesdicti however theirt???syted[]( ** awaitapp us
            "qwen2-0.5b-instruct-q2_k",     //Response:!!!!!!!
            "Qwen2-0.5B.Q2_K",              //Response:!!!!!!!
            "Qwen2-0.5B.Q3_K_L",            //Response:!!!!!!!
            "Qwen2-0.5B.Q3_K_M",            //Response:!!!!!!!
            "Qwen2-0.5B.Q3_K_S",            //Response:!!!!!!!
            "qwen2-0_5b-instruct-q4_k_m",   //Response:?  (????. - ( (g.-.
            "Qwen2-0.5B.Q5_1",              //Response:?????le?.arc?????box?? jihadist?? dyst late
            
            "Llama-3.2-1B-Instruct-Q4_K_M", //Response:  #\n\n\n## 3\n- and\n   - so, the
            
            "Qwen2.5-1.5B-Instruct-f16",    //Response:\n\n\n# # #\n#\n#\n\nimport sys\nsys.path
            "tinyllama-1.1b-chat-v1.0.Q8_0",    //Response:mathopmathopmathop ???? Luci LucyZygoteInitZygoteInitZygoteInitZygoteInitDataFrameDataFrameDataFrame? geldig
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",      //No Response:
            "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",   //Response:>\n\n---\n\n?>0ONENTdTdTdT>>ONENTONENT?ONENT
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", //No Response: Prior Response:Okay, the user just said "hello." I should respond in a friendly and      
            "qwen2.5-coder-3b-instruct-q8_0", //System.OutOfMemoryException
            "qwen2.5-coder-3b-instruct-q4_k_m", //System.OutOfMemoryException
            "qwen2.5-coder-3b-instruct-q2_k", //System.OutOfMemoryException
            ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt)
        {

            foreach (var m in Models)
            {
                Console.ForegroundColor = ConsoleColor.White;
                var returnedPrompt = false;
                var tok = 0;
                CancellationTokenSource cancellationTokenSource = new();
                var ggufPath = Path.Combine(ModelPath, $"{m}.gguf");
                if (!File.Exists(ggufPath)) continue;
                await Console.Out.WriteLineAsync($"Testing {m}");
                await Console.Out.FlushAsync();

                GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel();
                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                using var weights = GgufLoader.LoadWeightsToTransformerWeights(ggufPath, modelConfig);
                await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToTransformerWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                using var model = ModelFactory.CreateSession(weights, sharpConfig);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateSession executed in: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Stop();

                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                {
                    MaxTokens = 256,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                try
                {
                    var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    await Console.Out.WriteLineAsync();
                    await Console.Out.WriteLineAsync($"EXCEPTION: {ex.GetType().Name}: {ex.Message}");
                    await Console.Out.WriteLineAsync(ex.StackTrace?.Substring(0, 500) ?? "(no stack)");
                }

                async void Response(ChatStreamEntry text)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    await Console.Out.WriteAsync(text.Token);
                    tok++;
                    if (tok > 15) cancellationTokenSource.Cancel();
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
            await Console.Out.WriteLineAsync();
            await Console.Out.WriteLineAsync("Done!");
            Console.In.ReadLine();
        }
    }
}
