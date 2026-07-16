namespace SandBox.RunModels
{
    public class KnownFailingModels
    {
        private static readonly string[] Models =
        [

            "qwen2-0_5b-instruct-q4_k_m",   //Response:?  (????. - ( (g.-.
            "qwen2-0.5b-instruct-q2_k",     //Response:???u u performance?? 's elbowso??ef-.
            "Qwen2-0.5B.Q2_K",              //Response:performance sa?ero_ ?and i?.??f*+
            "Qwen2-0.5B.Q3_K_L",            //Response:./(?????????????????????
            "Qwen2-0.5B.Q3_K_M",            //Response:???\n\n??\_iple???\n\niez? manufactures gainedies
            "Qwen2-0.5B.Q3_K_S",            //Response:estring???emales?? je??? Gro  (??hibition Main
            
            "Qwen2-0.5B.Q5_1",              //Response:-+$?ergy=> Tw %(. Tw?????
 
            "Qwen2.5-1.5B-Instruct-f16",    //Response: ??,??????????????????????
            
            "Qwen3-0.6B-Q4_0",    //Responseonomyonomy??ultipart?ultipart?ultipart,,,itoito until until
            "Qwen3-0.6B-Q4_1",      //Response:;]/???? (nnenawai Holocaust ????? (waukeeentially???icuteenth?

                      
            "Llama-3.2-1B-Instruct-Q4_K_M", //Response:  and\nand\n\n a\n\n# "\na\n#  
            "tinyllama-1.1b-chat-v1.0.Q8_0",    //Response:\nDearlyrics
            "TinyLlama-1.1B-Chat-v1.0.Q4_K_M",  //Response:

            //https://huggingface.co/tensorblock/llama3-small-GGUF
            "llama3-small-Q2_K",//Response:))); bowInfo WATCH?? kho)."inky.Collectors essentially ???Events ?????? ?????? reactor
            "llama3-small-Q3_K_M", //Response:HostException savageInstrument??? ortaya.Process Hayes bleach?? bleach??RestController prefers,left_sleep
                      
            //https://huggingface.co/tensorblock/tiny-mistral-GGUF/tree/main
            "tiny-mistral-Q2_K", //Response:ilers0 vacuum?Kernel shedcontext glasses naked believe? Aquoffsetwhile dire
            "tiny-mistral-Q3_K_M", //Response:operated (' Beaut Gold? pocut spite viewing ihm materials WrazzreturnsHost
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
