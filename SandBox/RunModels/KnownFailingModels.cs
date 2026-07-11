using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;



namespace SandBox.RunModels
{
    public class KnownFailingModels
    {
        private static readonly string[] Models =
        [
            
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

            "Qwen2.5-1.5B-Instruct-f16",    //Response:??,?????????????????????

            "Qwen3-0.6B-Q4_0",    //Response: ( ( supplementuilder advancedhraductive??amahaDetachbatimi [{ulousISCO
            "Qwen3-0.6B-Q4_1",      //Response:;]/???? (nnenawai Holocaust ????? (waukeeentially???icuteenth?

            //Current Response: surfaced_into PICK quad&# embarrassing delay finale ????.Topic??? flex?\Validation
            //Best Known Response to date:Hello! I'm here. How can assist you?Hi what you can
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", 

            //Current Response:acias Const(md$(" reliability  Runtimelasmarapped Garlic '/'?? WaitFor?? Aer
            //Best Known Response to date:</think>\n\nHello! How Are You? ?? ?
            "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0",

            

                                          
            //Current Response:?? ???(vehicle???UniformLocation ? occupancy simplify generado?? Written prowess??
            //Best Known Response to date: \n\n\n# 1. Write a Python program to print the following pattern 
            "qwen2.5-1.5b-instruct-q8_0",

            

            "Llama-3.2-1B-Instruct-Q4_K_M", //Response:   and\nand\n\n a\n\n# "\na\n#     
            "tinyllama-1.1b-chat-v1.0.Q8_0",    //Response:L ??File mez ??Space veg Richard stepCAAFIGFile? Ord
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",  //Response: sacrificCAAFIG grat ?? Ord veg Richard step ..File Bayer ????????ifoliaars
            //https://huggingface.co/tensorblock/llama3-small-GGUF
            "llama3-small-Q2_K",//Response: "{\"_SIGNAL libido.viewDidLoad.createSequentialGroup Routing deprived ?????upp ?????rasing??? dogrudan_avatar
            "llama3-small-Q3_K",//Response:acias Const(md$(" reliability  Runtimelasmarapped Garlic '/'?? WaitFor?? Aer
                      
            //https://huggingface.co/tensorblock/tiny-mistral-GGUF/tree/main
            "tiny-mistral-Q2_K", //Response:idadections interactionsStub varianceGM Identity yards? tenant Icon":? pione???
            "tiny-mistral-Q3_K_M", //Response: typeof steep'' asksREEN sensors yeDDbritmost anywhere \;tabularreshold Grade
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
