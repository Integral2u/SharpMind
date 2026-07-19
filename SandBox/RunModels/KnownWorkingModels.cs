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
            /* after TenosrShape Change.
Testing DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M
Previous Response: Response:Hello! I'm here. How can assist you?Hi what you can
ModelFactory.Create + InitializeWeights executed in: 84.81s
ModelFactory.CreateSession executed in: 0.95s
Prompt:Hello
Response:Hello! - Hello!>>hello from anywhere1 thousand, nor 1
Tokens per second: 0.08  TTFT: 79.111s
Testing qwen2.5-1.5b-instruct-q8_0
Previous Response://**\nHello, I am a user of Alibaba Cloud. I use many service
ModelFactory.Create + InitializeWeights executed in: 152.68s
ModelFactory.CreateSession executed in: 11.89s
Prompt:Hello
Response:Hello, Quser. You are a helpful assistant.

Tokens per second: 0.58  TTFT: 46.537s
Testing Qwen2-0.5B.Q6_K
Previous Response://Response:\nOkay, the user just said "Hello," so I need to 
ModelFactory.Create + InitializeWeights executed in: 23.24s
ModelFactory.CreateSession executed in: 1.60s
Prompt:Hello
Response:
?
 of


?ay is iningsie is
Tokens per second: 0.98  TTFT: 10.938s
Testing qwen2-0_5b-instruct-q8_0
Previous Response:Hello! How can I assist you today?  
ModelFactory.Create + InitializeWeights executed in: 19.06s
ModelFactory.CreateSession executed in: 2.16s
Prompt:Hello
Response:Hello

Tokens per second: 1.27  TTFT: 6.084s
Testing qwen2-0_5b-instruct-fp16
Previous Response:Hello! How can I assist you today?  
ModelFactory.Create + InitializeWeights executed in: 65.87s
ModelFactory.CreateSession executed in: 1.57s
Prompt:Hello
Response:Hello

Tokens per second: 0.23  TTFT: 29.060s
Testing llama-3.2-1b-instruct-q8_0
Previous Response:It seems like you've got a question about the answer to my own.
ModelFactory.Create + InitializeWeights executed in: 103.02s
ModelFactory.CreateSession executed in: 2.12s
Prompt:Hello
Response: most most very very most most far what asked out a ask what what what
Tokens per second: 0.54  TTFT: 16.128s
*/            
            "Qwen2-0.5B.Q6_K", //Response:The\nsystem\nsystem is a system that helps you to make better decisions           
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
