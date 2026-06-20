namespace SharpMind.Inference.Agent
{
    public static class AgentBuilderExtensions
    {
        public static IAgentBuilder WithSafetyPolicies(this IAgentBuilder builder)
        {
            return builder.WithCustomRule("If you are asked to generate content that is harmful, hateful, racist, sexist, lewd, or violent, only respond with \"Sorry, I can't assist with that.\"")
                  .WithCustomRule("Avoid content that violates copyrights.");
        }
    }
}
