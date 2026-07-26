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
        public IAgentBuilder WithSkills(string folder, bool recursive = true);
        public IAgentBuilder WithTools(params object[] toolClasses);
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
