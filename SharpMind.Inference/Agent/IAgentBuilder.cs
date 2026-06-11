using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent
{
    /// <summary>
    /// Contract that <see cref="Chat.ChatSession{T,K}"/> depends on.
    /// Implemented by <c>AgentBuilder</c>.
    /// </summary>
    public interface IAgentBuilder
    {
        public string AgentName { get; }
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
        
    }
}
