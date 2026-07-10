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
            "SmolLM-135M.Q4_K_M",           //Response: ctypes initialization Program returnex Sw first customers sheBolplesswith Jointonies predicting
            "SmolLM2-135M-Instruct.Q4_K_M", //Response: indeerymourmereeno?emeteriesccordingquakescessionsrobelinedesidesohyd never $(
            "gemma-3-270m-it-Q8_0",     //Response: incessant Kisan Kisan agron poorest motorway Harareapples kilowattLife economists delivering motorway Highways intensive
            "gemma-3-270m-it-Q4_K_M",   //Response: harmonious?? cheap9 voic goalt llor Arbit privatisation cleats negotiatorsSpar recv justiciaHttpMethod
            "gemma-3-270m-it-F16"       //Response: incessant Kisan Kisan agron poorest HarareImprovingdegenerativeharmonic lousy motorway ProductivityapplesLife intensive                                                
        ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt) => await SharpMind.Samples.Examples.ModelListRunner.RunAsync(prompt, ModelPath, Models);
    }
}