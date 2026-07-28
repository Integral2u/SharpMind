namespace SandBox.RunModels
{
    public class KnownCoherentModels
    {
        public static readonly string[] Models = [.. AllQwenModels.Models.Where(p=>!p.StartsWith("Qwen2-0.5B.Q",StringComparison.InvariantCultureIgnoreCase)).Union(
        [
            //last working run 25/07/2026  regression new response:    Hello, dear friend.\n    I am here in the world of a little one.\n    I have to
            "llama-3.2-1b-instruct-q8_0",       //Response:It seems like you've got a question about the answer to my own.
            //last working run 24/07/2026 after commit 21906362 regression new response: I think you're notepad I'mostalgore I apologize for  i want to open to the  # i am using a\nThe answer was delayed in (t o s is not is located in u . The 2 1 i

            "SmolLM2-135M-Instruct.Q4_K_M", //Response:UserName: 10. You can you have been here to give advice and guidance for your friend is helpful! Your
           
            
        ]).Distinct()];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models, false);
    }
}
