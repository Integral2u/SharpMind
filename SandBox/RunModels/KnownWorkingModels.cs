using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;
using Xunit;

namespace SandBox.RunModels
{
    public class KnownWorkingModels
    {
        private static readonly string[] Models =
        [
            "Qwen3-0.6B-Q2_K",      //Response:?\nOkay, so I need to start with the user's message. 
            //last working run 25/07/2026  regression new response:    Hello, dear friend.\n    I am here in the world of a little one.\n    I have to
            "llama-3.2-1b-instruct-q8_0",       //Response:It seems like you've got a question about the answer to my own.
            //last working run 24/07/2026 after commit 21906362 regression new response: I think you're notepad I'mostalgore I apologize for  i want to open to the  # i am using a\nThe answer was delayed in (t o s is not is located in u . The 2 1 i

            "SmolLM2-135M-Instruct.Q4_K_M", //Response:UserName: 10. You can you have been here to give advice and guidance for your friend is helpful! Your
            "qwen2-0_5b-instruct-q4_k_m",   //Response:Hello! How can I assist you today?
            "qwen2-0.5b-instruct-q2_k",     //Response:Hello! How can I assist you today?

            "Qwen3-0.6B-Q4_0",      //Response:Okay, the user asked me to respond in a specific way.
            "Qwen3-0.6B-Q4_1",      //Response:Okay, the user just said "Hello and I need to respond    
            "qwen2-0_5b-instruct-q8_0",         //Response:Hello! How can I assist you today?
            "qwen2-0_5b-instruct-fp16",      //Response:Hello! How can I assist you today?  

            "Qwen3-0.6B-Q4_K_M",    //Response:\nOkay, the user just said "Hello," so I need to 

            "Qwen3-0.6B-Q5_K_M",    //Response:\nOkay, the user just said "Hello," so I need to                       
            "Qwen3-0.6B-Q3_K_M",    //Response:\nOkay, the user is asking for help with a problem. But
            "Qwen3-0.6B-Q6_K",      //Response:\nOkay, the user just said "Hello," so I need to            
 
            "Qwen3-0.6B-Q8_0",      //Response:\nOkay, the user just said "Hello," so I need to

            "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",//Response:Hi, thank you for asking your question. I'm just going through my memory again. Times ago, I have this busy
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", //Response:Hi! Welcome to Brain. I'm Trying To Teach Help '>>\n\nAlright, so I need help with this. The equation
            "qwen2.5-1.5b-instruct-q8_0", //Response:Hello! How can I help you today?
            "Qwen2.5-1.5B-Instruct-f16",    //Response:I am sorry for my mistake I did not understand your message correctly. Could you please rephrase the question or statement that you
            
            //Slow
            //"qwen2.5-coder-3b-instruct-q8_0", //Response: Hello! How can I assist you today?
            
            
        ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models, false);
    }
}
