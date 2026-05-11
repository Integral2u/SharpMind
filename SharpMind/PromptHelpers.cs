using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind
{
    public static class PromptHelpers
    {
        private static string? defaultSystemPrompt;
        public static string DefaultSystemPrompt => defaultSystemPrompt ??= GetEmbeddedPrompt("System.md");
        private static string? defaultAgentPrompt;
        public static string DefaultAgentPrompt => defaultAgentPrompt ??= GetEmbeddedPrompt("Agent.md");
        private static string GetEmbeddedPrompt(string name)
        {
            var assembly = Assembly.GetExecutingAssembly();
            string resourceName = $"{nameof(SharpMind)}.{name}";
            using var stream = assembly.GetManifestResourceStream(resourceName);
            if (stream == null) return string.Empty;
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd().Trim();
        }
    }
}
