using System;
using System.Collections.Generic;
using System.Text;

namespace SandBox.RunModels
{
    public class AllQwenModels
    {
        private static readonly string[] Models =
        [            
            "Qwen2-0.5B.Q2_K",
            "Qwen2-0.5B.Q3_K_L",
            "Qwen2-0.5B.Q3_K_M",
            "Qwen2-0.5B.Q3_K_S",
            "Qwen2-0.5B.Q5_1",
            "Qwen2-0.5B.Q6_K",
            "Qwen2-0.5B.Q8_0",

            "qwen2-0.5b-instruct-q2_k",         //Response:Hello! How can I assist you today?
            "qwen2-0_5b-instruct-q4_k_m",       //Response:Hello! How can I assist you today?            
            "qwen2-0_5b-instruct-q8_0",         //Response:Hello! How can I assist you today?
            "qwen2-0_5b-instruct-fp16",         //Response:Hello! How can I assist you today?  

            "qwen2.5-1.5b-instruct-q8_0",       //Response:Hello! How can I help you today?
            "Qwen2.5-1.5B-Instruct-f16",        //Response:I am sorry for my mistake I did not understand your message correctly. Could you please rephrase the question or statement that you
            
            "qwen2.5-coder-3b-instruct-q8_0",   //Response: Hello! How can I assist you today?
            "qwen2.5-coder-3b-instruct-q4_k_m", 
            "qwen2.5-coder-3b-instruct-q2_k",   
            
            "Qwen3-0.6B-Q2_K",                  //Response:?\nOkay, so I need to start with the user's message. 
            "Qwen3-0.6B-Q3_K_M",                //Response:\nOkay, the user is asking for help with a problem. But
            "Qwen3-0.6B-Q4_0",                  //Response:Okay, the user asked me to respond in a specific way.
            "Qwen3-0.6B-Q4_1",                  //Response:Okay, the user just said "Hello and I need to respond    
            "Qwen3-0.6B-Q4_K_M",                //Response:\nOkay, the user just said "Hello," so I need to 
            "Qwen3-0.6B-Q5_K_M",                //Response:\nOkay, the user just said "Hello," so I need to                                   
            "Qwen3-0.6B-Q6_K",                  //Response:\nOkay, the user just said "Hello," so I need to             
            "Qwen3-0.6B-Q8_0",                  //Response:\nOkay, the user just said "Hello," so I need to
            
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", //Response:Hi! Welcome to Brain. I'm Trying To Teach Help '>>\n\nAlright, so I need help with this. The equation
            "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",   //Response:Hi, thank you for asking your question. I'm just going through my memory again. Times ago, I have this busy

            ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt, bool diag)
        {
            if(diag) await DiagnosticModelRunner.RunAsync(prompt, Models);
            else await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
        }  
    }
}
