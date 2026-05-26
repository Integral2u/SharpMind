using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind
{
    /// <summary>
    /// Added to a class Method to identify a tool that can be used by the LLM.
    /// </summary>
    /// <param name="description">Description of the tool and how to use it.</param>
    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    public class AgentTool(string description) : Attribute
    {
        public string Description { get; init; } = description;
    }
    /// <summary>
    /// Method argument description.
    /// </summary>
    /// <param name="description">Desciption of this arguments purpose.</param>
    [AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
    public class AgentToolArgument(string description) : Attribute
    {
        public string Description { get; init; } = description;
    }
    public class EchoTool
    {
        [AgentTool("Use this if you want to simulate and echo in a canyon calling a persons name")]
        public string Echo([AgentToolArgument("The name to be returned twice")]string name) => $"{name} {name}";
        
        [AgentTool("Use this if you want to simulate and echo in a canyon calling a persons name")]
        public async Task<string> EchoAsync([AgentToolArgument("The name to be returned twice")] string name) => await Task.Run(() => Echo(name));
    }
}
