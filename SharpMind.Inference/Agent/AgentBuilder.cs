using JigSawDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind.Inference.Agent
{
    public class AgentBuilder(string agentName = "Delta")
    {
        public enum AgentSections
        {
            Tools
        }
        public string AgentName { get; init; } = agentName;

        public Dictionary<AgentSections, List<string>> Sections = [];
        /// <summary>
        /// Add tools from objects with defined <see cref="SharpMind.AgentTool.AgentTool(string)"/>
        /// </summary>
        /// <param name="toolClasses">Classes to get tools from</param>
        /// <returns></returns>
        public AgentBuilder WithTools(params object[] toolClasses)
        {
            static bool IsAwaitable(MethodInfo method)
            {
                var returnType = method.ReturnType;
                return returnType == typeof(Task) ||
                       (returnType.IsGenericType && returnType.GetGenericTypeDefinition() == typeof(Task<>)) ||
                       returnType == typeof(ValueTask) ||
                       (returnType.IsGenericType && returnType.GetGenericTypeDefinition() == typeof(ValueTask<>));
            }
            foreach (object toolClass in toolClasses)
            {
                if (toolClass is null) continue;
                var t = toolClass.GetType();
                if (!t.IsClass) continue;
                var tools = t.GetMethods().Where(m => m.GetCustomAttributes(typeof(AgentTool), true).Length != 0);
                if (tools == null) continue;
                foreach(var tool in tools)
                {
                    if (tool == null) continue;
                    if (tool.ReturnType == typeof(void)) continue;
                    var tda = tool.GetCustomAttributes(typeof(AgentTool), true).FirstOrDefault() as AgentTool;
                    if (tda == null || string.IsNullOrWhiteSpace(tda.Description)) continue;
                    
                    //foreach(var method in tool.GetParameters())
                    
                }
            }
            return this;
        }
        public string BuildSystemPrompt()
        {
            return string.Empty;
        }
        public string BuildAgentPrompt()
        {
            return string.Empty;
        }
    }
}
