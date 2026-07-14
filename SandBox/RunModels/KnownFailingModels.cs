namespace SandBox.RunModels
{
    public class KnownFailingModels
    {
        private static readonly string[] Models =
        [
            //Current Response: !!!!!!!
            //Best Known Response to date:Hello! I'm here. How can assist you?Hi what you can
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M", 

            //Know working Response:\nOkay, the user just said "Hello," so I need to
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q5_K_M",
            //Know working Response:\nOkay, the user just said "Hello," so I need to
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q4_K_M",
            //Know working Response:\Okay, the user is asking for help with a problem. But
            //Current Response:!!!!!!!
            "Qwen3-0.6B-Q3_K_M",

            "qwen2-0_5b-instruct-q4_k_m",   //Response:!!!!!!!
            "qwen2-0.5b-instruct-q2_k",     //Response:???u u performance?? 's elbowso??ef-.
            "Qwen2-0.5B.Q2_K",              //Response:performance sa?ero_ ?and i?.??f*+
            "Qwen2-0.5B.Q3_K_L",            //Response:./(?????????????????????
            "Qwen2-0.5B.Q3_K_M",            //Response:!!!!!!!
            "Qwen2-0.5B.Q3_K_S",            //Response:estring???emales?? je??? Gro  (??hibition Main
            
            "Qwen2-0.5B.Q5_1",              //Response:-+$?ergy=> Tw %(. Tw?????
 
            

            "Qwen2.5-1.5B-Instruct-f16",    //Response: ??,??????????????????????
            
            "Qwen3-0.6B-Q4_0",    //Response:!!!!!!!
            "Qwen3-0.6B-Q4_1",      //Response:;].ToDouble'postouce */)EMPL     Meering }])\n Tangounistillis? -\n\n Oblysize

            

                                          
            "Llama-3.2-1B-Instruct-Q4_K_M", //Response:!!!!!!!    
            "tinyllama-1.1b-chat-v1.0.Q8_0",    //Response:\nDearlyrics
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",  //(currenlty seems to hang)Response: sacrificCAAFIG grat ?? Ord veg Richard step ..File Bayer ????????ifoliaars

            //https://huggingface.co/tensorblock/llama3-small-GGUF
            "llama3-small-Q2_K",//Response:?? nær??? dataSize)c PittTranslatef.columnHeader.columnHeader beaut helpersreturns yilindasad Brad
            "llama3-small-Q3_K_M", //Response: ??')}}" UNKNOWN plaintiffs prevalentonic Ca získal/select Latina discovered lubric premier limitlessaccount
                      
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
