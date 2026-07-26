using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SandBox.RunModels
{
    public class KnownGiberishModels
    {
        private static readonly string[] Models =
        [            
            //"SmolLM-135M.Q4_K_M",           //Response:\n\n0. 125639784
            
            "gemma-3-270m-it-Q8_0",     //Response: incessant Kisan Kisan agron poorest motorway Harareapples kilowattLife economists delivering motorway Highways intensive
            //"gemma-3-270m-it-Q4_K_M",   //Response: incessant lousy intensive?? motorway Kisan agric prizeddegenerative?pute Kisanharmonic Harare Precious
            //"gemma-3-270m-it-F16"       //Response: incessant Kisan Kisan agron poorest HarareImprovingdegenerativeharmonic lousy motorway ProductivityapplesLife intensive                                                
        ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
    }
}