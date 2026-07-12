using ILGPU.IR.Values;
using ILGPU.Runtime;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;
using static System.Runtime.InteropServices.JavaScript.JSType;



namespace SandBox.RunModels
{
    public class KnownFailingModels
    {
        private static readonly string[] Models =
        [
            //"TinyLlama-1.1B-Chat-v1.0.Q4_K_M",
            /*Exception
            Response:Fatal error.
System.AccessViolationException: Attempted to read or write protected memory. This is often an indication that other memory is corrupt.
   at SharpMind.Core.Quantization.QuantizationKernels.VecDotQ6K_FMA(Single*, Byte*, Int32, Int32)
   at SharpMind.Core.Quantization.QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA(Single*, Byte*, Single*, Int32, Int32, Int32)
   at InferenceLinearLayer7043582.QuantizedMatMulFn(Single*, Byte*, Single*, Int32, Int32, Int32)
   at SharpMind.Model.Layers.InferenceLinearLayer.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.FfnKernels.Gated(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.Layers.LinearLayer, SharpMind.Model.Layers.LinearLayer, SharpMind.Core.Activations.ActivationOps, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.GatedFfnLayer.ApplyFfn(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.FfnLayer.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.UnhookedTransformerBlock.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.IKVCache, Int32, Boolean, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Arch.DecoderArch.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.IKVCache[], Int32, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Transformer.ForwardLastLogits(SharpMind.Core.Tensors.Tensor`1<Int32>, SharpMind.Model.IKVCache[], Int32, SharpMind.Core.Memory.Workspace)
   at SharpMind.Inference.StandardGenerator`1+<GenerateFromTokensAsync>d__13[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]](System.__Canon ByRef)
   at SharpMind.Inference.StandardGenerator`1+<GenerateFromTokensAsync>d__13[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].System.Collections.Generic.IAsyncEnumerator<System.String>.MoveNextAsync()
   at SharpMind.Inference.Chat.ChatSession`2+<GetResponseStreamAsync>d__96[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]](System.__Canon ByRef)
   at SharpMind.Inference.Chat.ChatSession`2+<GetResponseStreamAsync>d__96[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].System.Collections.Generic.IAsyncEnumerator<SharpMind.Inference.Chat.ChatStreamEntry>.MoveNextAsync()
   at SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]], SharpMind.Inference, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<StartChatAsync>d__97<System.__Canon,System.__Canon> ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder`1[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].Start[[SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]], SharpMind.Inference, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<StartChatAsync>d__97<System.__Canon,System.__Canon> ByRef)
   at SharpMind.Inference.Chat.ChatSession`2[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].StartChatAsync(System.Func`1<System.Threading.Tasks.Task`1<SharpMind.Inference.Chat.ChatMessage>>, System.Action`1<SharpMind.Inference.Chat.ChatStreamEntry>, System.Threading.CancellationToken)
   at SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0, SharpMind.Samples, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__0 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0, SharpMind.Samples, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__0 ByRef)
   at SharpMind.Samples.Examples.ModelListRunner.RunAsync(System.String, System.String, System.String[], Boolean)
   at SandBox.RunModels.KnownFailingModels+<RunAsync>d__2.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SandBox.RunModels.KnownFailingModels+<RunAsync>d__2, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__2 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[SandBox.RunModels.KnownFailingModels+<RunAsync>d__2, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__2 ByRef)
   at SandBox.RunModels.KnownFailingModels.RunAsync(System.String)
   at Program+<<Main>$>d__0.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[Program+<<Main>$>d__0, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<<Main>$>d__0 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[Program+<<Main>$>d__0, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<<Main>$>d__0 ByRef)
   at Program.<Main>$(System.String[])
   at Program.<Main>(System.String[])
            //"llama3-small-Q3_K",
            /*Exception
            Response:Fatal error.
System.AccessViolationException: Attempted to read or write protected memory. This is often an indication that other memory is corrupt.
   at SharpMind.Core.Quantization.QuantizationKernels.VecDotQ6K_FMA(Single*, Byte*, Int32, Int32)
   at SharpMind.Core.Quantization.QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA(Single*, Byte*, Single*, Int32, Int32, Int32)
   at InferenceLinearLayer24457708.QuantizedMatMulFn(Single*, Byte*, Single*, Int32, Int32, Int32)
   at SharpMind.Model.Layers.InferenceLinearLayer.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.FfnKernels.Gated(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.Layers.LinearLayer, SharpMind.Model.Layers.LinearLayer, SharpMind.Core.Activations.ActivationOps, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.GatedFfnLayer.ApplyFfn(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.Ffn.FfnLayer.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Layers.UnhookedTransformerBlock.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.IKVCache, Int32, Boolean, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Arch.DecoderArch.Forward(SharpMind.Core.Tensors.Tensor`1<Single>, SharpMind.Model.IKVCache[], Int32, SharpMind.Core.Memory.Workspace)
   at SharpMind.Model.Transformer.ForwardLastLogits(SharpMind.Core.Tensors.Tensor`1<Int32>, SharpMind.Model.IKVCache[], Int32, SharpMind.Core.Memory.Workspace)
   at SharpMind.Inference.StandardGenerator`1+<GenerateFromTokensAsync>d__13[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]](System.__Canon ByRef)
   at SharpMind.Inference.StandardGenerator`1+<GenerateFromTokensAsync>d__13[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].System.Collections.Generic.IAsyncEnumerator<System.String>.MoveNextAsync()
   at SharpMind.Inference.Chat.ChatSession`2+<GetResponseStreamAsync>d__96[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]](System.__Canon ByRef)
   at SharpMind.Inference.Chat.ChatSession`2+<GetResponseStreamAsync>d__96[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].System.Collections.Generic.IAsyncEnumerator<SharpMind.Inference.Chat.ChatStreamEntry>.MoveNextAsync()
   at SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]], SharpMind.Inference, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<StartChatAsync>d__97<System.__Canon,System.__Canon> ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder`1[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].Start[[SharpMind.Inference.Chat.ChatSession`2+<StartChatAsync>d__97[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]], SharpMind.Inference, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<StartChatAsync>d__97<System.__Canon,System.__Canon> ByRef)
   at SharpMind.Inference.Chat.ChatSession`2[[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e],[System.__Canon, System.Private.CoreLib, Version=10.0.0.0, Culture=neutral, PublicKeyToken=7cec85d7bea7798e]].StartChatAsync(System.Func`1<System.Threading.Tasks.Task`1<SharpMind.Inference.Chat.ChatMessage>>, System.Action`1<SharpMind.Inference.Chat.ChatStreamEntry>, System.Threading.CancellationToken)
   at SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0, SharpMind.Samples, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__0 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[SharpMind.Samples.Examples.ModelListRunner+<RunAsync>d__0, SharpMind.Samples, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__0 ByRef)
   at SharpMind.Samples.Examples.ModelListRunner.RunAsync(System.String, System.String, System.String[], Boolean)
   at SandBox.RunModels.KnownFailingModels+<RunAsync>d__2.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[SandBox.RunModels.KnownFailingModels+<RunAsync>d__2, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__2 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[SandBox.RunModels.KnownFailingModels+<RunAsync>d__2, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<RunAsync>d__2 ByRef)
   at SandBox.RunModels.KnownFailingModels.RunAsync(System.String)
   at Program+<<Main>$>d__0.MoveNext()
   at System.Runtime.CompilerServices.AsyncMethodBuilderCore.Start[[Program+<<Main>$>d__0, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<<Main>$>d__0 ByRef)
   at System.Runtime.CompilerServices.AsyncTaskMethodBuilder.Start[[Program+<<Main>$>d__0, SandBox, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null]](<<Main>$>d__0 ByRef)
   at Program.<Main>$(System.String[])
   at Program.<Main>(System.String[])
            */
            "qwen2-0_5b-instruct-q4_k_m",   //Response:!!!!!!!
            "qwen2-0.5b-instruct-q2_k",     //Response:???u u performance?? 's elbowso??ef-.
            "Qwen2-0.5B.Q2_K",              //Response:performance sa?ero_ ?and i?.??f*+
            "Qwen2-0.5B.Q3_K_L",            //Response:./(?????????????????????
            "Qwen2-0.5B.Q3_K_M",            //Response:!!!!!!!
            "Qwen2-0.5B.Q3_K_S",            //Response:estring???emales?? je??? Gro  (??hibition Main
            
            "Qwen2-0.5B.Q5_1",              //Response:-+$?ergy=> Tw %(. Tw?????
 
            //Know working Response:The\nsystem\nsystem is a system that helps you to make better decisions
            //Current Response:5??,?? (5? combined?- . -__??
            "Qwen2-0.5B.Q6_K",

            //Know working Response:\nOkay, the user just said "Hello," so I need to
            //Current Response: libertinechein dancesissionabelle???esadamenteotropic oficial IonicPage(filequalTo=back disappointed
            "Qwen3-0.6B-Q6_K",

            //Know working Response:\nOkay, the user just said "Hello," so I need to
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q5_K_M",
            //Know working Response:\nOkay, the user just said "Hello," so I need to
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q4_K_M",
            //Know working Response:\Okay, the user is asking for help with a problem. But
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q3_K_M",

             //Know working Response:nOkay, so I need to start with the user's message.
            //Current Response:/Getty WikiLeaks??owardrylicemensemens shedding
            "Qwen3-0.6B-Q2_K",   

            "Qwen2.5-1.5B-Instruct-f16",    //Response: ??,??????????????????????
            
            "Qwen3-0.6B-Q4_0",    //Response:!!!!!!!
            "Qwen3-0.6B-Q4_1",      //Response:;].ToDouble'postouce */)EMPL     Meering }])\n Tangounistillis? -\n\n Oblysize

            //Current Response: !!!!!!!
            //Best Known Response to date:Hello! I'm here. How can assist you?Hi what you can
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", 

                                          
            "Llama-3.2-1B-Instruct-Q4_K_M", //Response:!!!!!!!    
            "tinyllama-1.1b-chat-v1.0.Q8_0",    //Response:\nDearlyrics
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",  //Response: sacrificCAAFIG grat ?? Ord veg Richard step ..File Bayer ????????ifoliaars
            
            //https://huggingface.co/tensorblock/llama3-small-GGUF
            "llama3-small-Q2_K",//Response:?? nær??? dataSize)c PittTranslatef.columnHeader.columnHeader beaut helpersreturns yilindasad Brad
            
                      
            //https://huggingface.co/tensorblock/tiny-mistral-GGUF/tree/main
            "tiny-mistral-Q2_K", //Response: reassiled? Cameronaltern muy intellig Crim extentrowned proposals COP mascul mobile scratch
            "tiny-mistral-Q3_K_M", //Response:hw<>( traders awkwardexper? procedITlow Trump underc?wX performing?
            /*
            //Unknown shape
            //https://huggingface.co/prism-ml/Bonsai-8B-gguf            
            "Bonsai-8B",    //Out of Memory
            "Bonsai-8B-Q1_0",   //Out of Memory
            
            //"Phi-3-mini-4k-instruct-q4", //System.OutOfMemoryException try cached loader
            //"qwen2.5-coder-3b-instruct-q8_0", //System.OutOfMemoryException
            //"qwen2.5-coder-3b-instruct-q4_k_m", //System.OutOfMemoryException
            //"qwen2.5-coder-3b-instruct-q2_k", //System.OutOfMemoryException
            */
            ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
    }
}
