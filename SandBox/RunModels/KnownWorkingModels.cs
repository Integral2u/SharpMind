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
                    
            "qwen2-0_5b-instruct-q8_0",         //Response:Hello! How can I assist you today?
            "qwen2-0_5b-instruct-fp16",      //Response:Hello! How can I assist you today?  

            "Qwen3-0.6B-Q4_K_M",    //Response:\nOkay, the user just said "Hello," so I need to 

            "Qwen3-0.6B-Q5_K_M",    //Response:\nOkay, the user just said "Hello," so I need to                       
            "Qwen3-0.6B-Q3_K_M",    //Response:\nOkay, the user is asking for help with a problem. But
            "Qwen3-0.6B-Q6_K",      //Response:\nOkay, the user just said "Hello," so I need to            
            "Qwen3-0.6B-Q2_K",      //Response:?\nOkay, so I need to start with the user's message.  
            "Qwen3-0.6B-Q8_0",      //Response:\nOkay, the user just said "Hello," so I need to

            "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",//Response:Hello! How Are You? ?? ?
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", //Response:Hello! I'm here. How can assist you?Hi what you can
            "qwen2.5-1.5b-instruct-q8_0", //Response: //**\nHello, I am a user of Alibaba Cloud. I use many service

            "llama-3.2-1b-instruct-q8_0",       //Response:It seems like you've got a question about the answer to my own.
            

            
            
        ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models, false);
    }
}
