namespace SharpMind.Core.AgentTools
{
   
    public class EchoTool
    {
        [ToolDesc("Simulates a canyon echo by repeating a person's name.")]
        public static string Echo([ToolDesc("The name to echo back twice.")]string name) => $"{name} {name}";
        
        [ToolDesc("Async canyon echo — same as Echo but non-blocking.")]
        public static async Task<string> EchoAsync([ToolDesc("The name to echo back twice.")] string name) => await Task.Run(() => Echo(name));
    }
}
