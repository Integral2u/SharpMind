using System;

namespace SharpMind
{
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Parameter)]
    public sealed class ToolDescAttribute(string text) : Attribute
    {
        public string Text { get; } = text;
    }
   
    public class EchoTool
    {
        [ToolDesc("Simulates a canyon echo by repeating a person's name.")]
        public string Echo([ToolDesc("The name to echo back twice.")]string name) => $"{name} {name}";
        
        [ToolDesc("Async canyon echo — same as Echo but non-blocking.")]
        public async System.Threading.Tasks.Task<string> EchoAsync([ToolDesc("The name to echo back twice.")] string name) => await System.Threading.Tasks.Task.Run(() => Echo(name));
    }
}
