using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SandBox.RunModels
{
    public class KnownWorkingModels
    {
        private static readonly string[] Models =
        [         

            //Possible responce chat formatting issues
            "Qwen2-0.5B.Q6_K",                  //Response:The\nsystem\nsystem is a system that helps you to make better decisions

            //Working                                     
            "llama-3.2-1b-instruct-q8_0",       //Response:It seems like you've got a question about the answer to my own.            
            "qwen2-0_5b-instruct-q8_0",         //Response:Hello! How can I assist you today?
            "qwen2-0_5b-instruct-fp16",      //Response:Hello! How can I assist you today?                                               
            "Qwen3-0.6B-Q8_0",      //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q6_K",      //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q5_K_M",    //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q4_K_M",    //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q3_K_M",    //Response:<think>\Okay, the user is asking for help with a problem. But
            "Qwen3-0.6B-Q2_K",      //Response:?\nOkay, so I need to start with the user's message.
        ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
    }
}
