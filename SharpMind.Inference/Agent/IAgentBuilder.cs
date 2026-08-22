using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Inference.Chat;

namespace SharpMind.Inference.Agent
{
    /// <summary>
    /// Contract that <see cref="Chat.ChatSession{T,K}"/> depends on.
    /// Implemented by <c>AgentBuilder</c>.
    /// </summary>
    public interface IAgentBuilder
    {
        public string AgentName { get; }
        public HashSet<string> DisabledTools { get; set; }
        public IReadOnlyList<string> RegisteredToolNames { get; }
        public IContextCompactor? Compactor { get; }
        public IReadOnlyList<IContextCompactor> PluginCompactors { get; }
        public IReadOnlyList<IPromptPreProcessor> PluginPreProcessors { get; }
        public IReadOnlyList<IPromptPostProcessor> PluginPostProcessors { get; }
        public IAgentBuilder WithCustomBehavior(string behavior);
        public IAgentBuilder WithCustomRule(string rule);
        public IAgentBuilder WithSkill(string file);
        public IAgentBuilder WithSkillContent(string content);
        public IAgentBuilder WithAdditionalSystemPrompt(string prompt);
        public IAgentBuilder WithSkills(string folder, bool recursive = true);
        /// <summary>Standalone system prompts inserted at the top of the history, before the synthesized agent prompt.</summary>
        public IReadOnlyList<string> AdditionalSystemPrompts { get; }
        public IAgentBuilder WithTools(params object[] toolClasses);

        /// <summary>
        /// Registers a delegate-based tool with an explicit JSON-schema definition.
        /// Unlike <see cref="WithTools"/>, which reflects over <c>[ToolDesc]</c>
        /// attributes, this overload is designed for external callers (e.g. the
        /// Microsoft.Extensions.AI adapter) that supply their own tool metadata
        /// and invocation logic.
        /// </summary>
        /// <param name="name">Tool name the model will emit in JSON.</param>
        /// <param name="description">Human-readable description for the agent prompt.</param>
        /// <param name="schema">
        /// A JSON Schema <c>parameters</c> object describing the tool's arguments.
        /// Example: <c>{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}</c>.
        /// </param>
        /// <param name="execute">
        /// Async delegate invoked with the parsed <c>arguments</c> object.
        /// Must return a string result for the model.
        /// </param>
        IAgentBuilder WithTool(string name, string description, JsonObject schema, Func<JsonObject, Task<string>> execute);
        /// <summary>Builds the system prompt text for the current agent configuration.</summary>
        public string BuildAgentPrompt();
        /// <summary>
        /// Dispatches a tool call described by <paramref name="toolCall"/> and returns
        /// a <c>{ "status": "success"|"error", "data"|"message": "..." }</c> result object.
        /// </summary>
        /// <param name="toolCall">
        /// A JSON object with at minimum a <c>tool</c> string field and an
        /// <c>arguments</c> object field, as produced by the model.
        /// </param>
        public Task<JsonObject> CallToolAsync(JsonObject toolCall);

        /// <summary>
        /// Creates and registers a sub-agent that can be called by the model
        /// via the <c>{{agent:Name[:temp=X][:seed=Y]:query}}</c> format.
        /// </summary>
        public IAgent CreateAgent(AgentConfig config);

        /// <summary>All registered sub-agents, keyed by their auto/assigned name.</summary>
        public IReadOnlyDictionary<string, IAgent> RegisteredAgents { get; }

        /// <summary>Whether sub-agent delegation is enabled (opt-in via <see cref="WithAgents"/>).</summary>
        public bool AgentsEnabled { get; }

        /// <summary>Maximum sub-agent nesting depth when agents are enabled.</summary>
        public int MaxAgentDepth { get; }

        /// <summary>Enables sub-agent delegation with the given nesting depth.</summary>
        public IAgentBuilder WithAgents(int depth = 2);
    }
}
