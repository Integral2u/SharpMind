using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Inference.Chat;

namespace SharpMind.Inference.Agent
{
    public class AgentBuilder(string agentName = "Delta", SamplingConfig? samplingConfig = null) : IAgentBuilder
    {
        public enum AgentSections
        {
            Role,
            Rules,
            Behavior,
            Tools,
            ToolCallFormat,
            Skills
        }
        public string AgentName { get; init; } = agentName;
        public SamplingConfig SamplingConfig { get; init; } = samplingConfig ?? new();
        public IContextCompactor? Compactor { get; set; }
        public IReadOnlyList<IContextCompactor> PluginCompactors { get; set; } = [];
        public IReadOnlyList<IPromptPreProcessor> PluginPreProcessors { get; set; } = [];
        public IReadOnlyList<IPromptPostProcessor> PluginPostProcessors { get; set; } = [];
        public HashSet<string> DisabledTools { get; set; } = [];

        public IReadOnlyList<string> RegisteredToolNames => ToolMethods.Keys.ToList();

        // Not currently used in prompt building but available for callers that want to
        // inspect or manipulate sections as keyed lists.
        public Dictionary<AgentSections, List<string>> Sections = [];
        private readonly Dictionary<string, (MethodInfo Method, object Instance)> ToolMethods = [];
        public readonly JsonArray ToolDefinitions = [];

        public readonly List<string> Behaviors = [];
        public readonly List<string> Skills = [];
        public readonly List<string> Rules = [];

        // Sub-agent registry
        private readonly Dictionary<string, IAgent> _agents = [];
        private int _unnamedCounter;
        public IReadOnlyDictionary<string, IAgent> RegisteredAgents => _agents;

        private bool _agentsEnabled;
        private int _maxAgentDepth = 2;
        public bool AgentsEnabled => _agentsEnabled;
        public int MaxAgentDepth => _maxAgentDepth;

        // Builder helpers

        /// <summary>Adds a free-form behavioral instruction (idempotent).</summary>
        public IAgentBuilder WithCustomBehavior(string behavior)
        {
            // FIX: was "if (Contains)" — only added duplicates, never new entries
            if (!Behaviors.Contains(behavior))
                Behaviors.Add(behavior);
            return this;
        }

        /// <summary>Adds a rule (idempotent).</summary>
        public IAgentBuilder WithCustomRule(string rule)
        {
            // FIX: same inverted guard as above
            if (!Rules.Contains(rule))
                Rules.Add(rule);
            return this;
        }

        /// <summary>
        /// Loads a single skill file (.md / .txt) and appends its content to the Skills section.
        /// Silently skips if the file does not exist or is empty.
        /// </summary>
        public IAgentBuilder WithSkill(string file)
        {
            if (string.IsNullOrWhiteSpace(file) || !File.Exists(file))
                return this;

            var content = File.ReadAllText(file).Trim();
            if (content.Length == 0)
                return this;

            // Avoid loading the same physical file twice
            if (!Skills.Contains(content))
                Skills.Add(content);

            return this;
        }

        /// <summary>
        /// Walks <paramref name="folder"/> for files named <c>skill.md</c> or <c>skills.md</c>
        /// (case-insensitive) and loads each one. Recurses into sub-directories when
        /// <paramref name="recursive"/> is <see langword="true"/> (default).
        /// Silently skips missing or empty folders.
        /// </summary>
        public IAgentBuilder WithSkills(string folder, bool recursive = true)
        {
            if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
                return this;

            var option = recursive
                ? SearchOption.AllDirectories
                : SearchOption.TopDirectoryOnly;

            // Accept both "skill.md" and "skills.md" conventions
            var skillFiles = Directory
                .EnumerateFiles(folder, "*.md", option)
                .Where(f =>
                {
                    var name = Path.GetFileNameWithoutExtension(f);
                    return string.Equals(name, "skill", StringComparison.OrdinalIgnoreCase)
                        || string.Equals(name, "skills", StringComparison.OrdinalIgnoreCase);
                })
                .OrderBy(f => f); // deterministic ordering

            foreach (var file in skillFiles)
                WithSkill(file);

            return this;
        }

        /// <summary>
        /// Reflects over <paramref name="toolClasses"/> and registers every public method
        /// decorated with <see cref="ToolDescAttribute"/> whose parameters are also all
        /// decorated with <see cref="ToolDescAttribute"/>.
        /// Will silently skip methods that are void, non-returning Tasks, have un-annotated
        /// parameters, or whose name collides with an already-registered tool.
        /// </summary>
        public IAgentBuilder WithTools(params object[] toolClasses)
        {
            foreach (object toolClass in toolClasses)
            {
                if (toolClass is null) continue;
                var t = toolClass.GetType();
                if (!t.IsClass) continue;

                var tools = t.GetMethods()
                             .Where(m => m.GetCustomAttributes(typeof(ToolDescAttribute), true).Length != 0);

                foreach (var tool in tools)
                {
                    if (tool is null) continue;
                    if (tool.ReturnType == typeof(void)) continue;
                    if (tool.ReturnType == typeof(Task)) continue;
                    if (ToolMethods.ContainsKey(tool.Name)) continue;
                    if (DisabledTools.Contains(tool.Name)) continue;

                    // All parameters must carry [ToolDesc]
                    var missingAnnotations = tool.GetParameters()
                        .Where(p => p.GetCustomAttribute<ToolDescAttribute>() is null)
                        .Select(p => p.Name)
                        .ToList();

                    if (missingAnnotations.Count > 0) continue;

                    ToolMethods.Add(tool.Name, (tool, toolClass));
                    ToolDefinitions.Add(BuildToolDef(tool));
                }
            }
            return this;
        }

        // Tool definition builders

        private static JsonObject BuildToolDef(MethodInfo method)
        {
            var desc = method.GetCustomAttribute<ToolDescAttribute>()?.Text ?? "";
            var ctx = new NullabilityInfoContext();

            var props = new JsonObject();
            var required = new List<string>();

            foreach (var p in method.GetParameters())
            {
                props[p.Name!] = BuildParamSchema(p);

                bool isOptional = p.HasDefaultValue
                               || ctx.Create(p).WriteState == NullabilityState.Nullable;

                if (!isOptional) required.Add(p.Name!);
            }

            return new JsonObject
            {
                ["name"] = method.Name,
                ["description"] = desc,
                ["parameters"] = new JsonObject
                {
                    ["type"] = "object",
                    ["properties"] = props,
                    ["required"] = new JsonArray([.. required.Select(r => JsonValue.Create(r))])
                }
            };
        }

        private static JsonObject BuildParamSchema(ParameterInfo param)
        {
            var desc = param.GetCustomAttribute<ToolDescAttribute>()?.Text ?? "";
            var schema = JsonTypeToSchema(param.ParameterType);

            schema["description"] = desc;

            if (param.HasDefaultValue && param.DefaultValue is not null)
                schema["default"] = JsonValue.Create(param.DefaultValue.ToString());

            return schema;
        }

        private static JsonObject JsonTypeToSchema(Type type)
        {
            type = Nullable.GetUnderlyingType(type) ?? type;

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Task<>))
                type = type.GetGenericArguments()[0];

            if (type == typeof(string)) return Typed("string");
            if (type == typeof(bool)) return Typed("boolean");
            if (type == typeof(int) || type == typeof(long) || type == typeof(short)) return Typed("integer");
            if (type == typeof(float) || type == typeof(double) || type == typeof(decimal)) return Typed("number");
            if (type.IsArray || IsGenericList(type)) return Typed("array");

            return Typed("object");
        }

        private static bool IsGenericList(Type t) =>
            t.IsGenericType && t.GetGenericTypeDefinition() == typeof(List<>);

        private static JsonObject Typed(string type) => new() { ["type"] = type };

        // Sub-agent registration

        /// <summary>
        /// Creates and registers a sub-agent. When <paramref name="config.Name"/> is null,
        /// an auto-name is generated using the Greek tier system
        /// (<c>{Deity}-{Tier}</c> e.g. <c>Athena-Alpha</c>).
        /// The agent is callable by the model via <c>{{agent:Name:query}}</c>.
        /// </summary>
        /// <summary>Enables sub-agent delegation with the given nesting depth.</summary>
        public IAgentBuilder WithAgents(int depth = 2)
        {
            _agentsEnabled = true;
            _maxAgentDepth = depth;
            return this;
        }

        /// <summary>
        /// Creates and registers a sub-agent. When <paramref name="config.Name"/> is null,
        /// an auto-name is generated using the Greek tier system
        /// (<c>{Deity}-{Tier}</c> e.g. <c>Athena-Alpha</c>).
        /// The agent is callable by the model via <c>{{agent:Name:query}}</c>.
        /// </summary>
        public IAgent CreateAgent(AgentConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);
            ArgumentException.ThrowIfNullOrWhiteSpace(config.SystemPrompt);

            string? name = config.Name;
            if (string.IsNullOrWhiteSpace(name))
            {
                name = GreekTier.AutoName(config.Temperature, ref _unnamedCounter, [.. _agents.Keys]);
            }

            if (_agents.ContainsKey(name))
                throw new InvalidOperationException($"An agent named '{name}' is already registered.");

            var agent = new Agent(name, config);
            _agents[name] = agent;
            return agent;
        }

        // Tool invocation
        /// <summary>
        /// Dispatches a tool call from the model's JSON response.
        /// <paramref name="toolCall"/> must be a JSON object with a <c>tool</c>
        /// string field and an <c>arguments</c> object field — exactly the shape
        /// produced by the agent prompt's Tool Call Format section.
        /// </summary>
        public async Task<JsonObject> CallToolAsync(JsonObject toolCall)
        {
            try
            {
                var toolName = toolCall["tool"]?.GetValue<string>()
                    ?? throw new ArgumentException("Missing required field: 'tool'.");

                if (!ToolMethods.TryGetValue(toolName, out var entry))
                    throw new ArgumentException($"Unknown tool: '{toolName}'.");

                var args = toolCall["arguments"]?.AsObject() ?? [];
                var invokeArgs = BindArguments(entry.Method, args);

                var raw = entry.Method.Invoke(entry.Instance, invokeArgs)
                    ?? throw new InvalidOperationException("Tool returned null.");

                var data = raw switch
                {
                    Task<string> t => await t,
                    Task<int> t => (await t).ToString(),
                    Task<object> t => (await t).ToString()!,
                    Task t => await t.ContinueWith(_ => ""),
                    _ => raw.ToString()!
                };

                return Success(data!);
            }
            catch (TargetInvocationException ex) { return Error(ex.InnerException?.Message ?? ex.Message); }
            catch (Exception ex) { return Error(ex.Message); }
        }


        private static object?[] BindArguments(MethodInfo method, JsonObject args)
        {
            var ctx = new NullabilityInfoContext();
            var @params = method.GetParameters();
            var result = new object?[@params.Length];

            for (int i = 0; i < @params.Length; i++)
            {
                var p = @params[i];
                var node = args[p.Name!];

                if (node is null)
                {
                    if (p.HasDefaultValue)
                    { result[i] = p.DefaultValue; continue; }
                    if (ctx.Create(p).WriteState == NullabilityState.Nullable)
                    { result[i] = null; continue; }
                    throw new ArgumentException($"Required argument '{p.Name}' is missing.");
                }

                result[i] = CoerceValue(node, p.ParameterType);
            }

            return result;
        }

        private static object? CoerceValue(JsonNode node, Type targetType)
        {
            targetType = Nullable.GetUnderlyingType(targetType) ?? targetType;

            return targetType switch
            {
                _ when targetType == typeof(string) => node.GetValue<string>(),
                _ when targetType == typeof(int) => node.GetValue<int>(),
                _ when targetType == typeof(long) => node.GetValue<long>(),
                _ when targetType == typeof(float) => node.GetValue<float>(),
                _ when targetType == typeof(double) => node.GetValue<double>(),
                _ when targetType == typeof(decimal) => node.GetValue<decimal>(),
                _ when targetType == typeof(bool) => node.GetValue<bool>(),
                _ => node.Deserialize(targetType)
            };
        }

        private static JsonObject Success(string data) => new() { ["status"] = "success", ["data"] = data };
        private static JsonObject Error(string message) => new() { ["status"] = "error", ["message"] = message };
        private static string TemperaturePersonality(float temperature) => temperature switch
        {
            <= 0.1f => "exacting and strictly literal",
            <= 0.3f => "methodical and analytical",
            <= 0.5f => "pragmatic and measured",
            <= 0.7f => "thoughtful and adaptive",
            <= 0.9f => "imaginative and expressive",
            <= 1.1f => "creative and exploratory",
            _ => "unconventional and abstract"
        };


        // Prompt building

        /// <summary>
        /// Builds a plain-text system prompt from the current builder state.
        /// The result is architecture-agnostic markdown; the caller (e.g. ChatSession
        /// via <c>IChatPromptFormatter</c>) is responsible for wrapping it in whatever
        /// special tokens the model requires (ChatML, Llama-3, Mistral, Phi-3, etc.).
        ///
        /// Usage from ChatSession:
        /// <code>
        ///   session.AddMessage(ChatRole.System, agent.BuildAgentPrompt());
        /// </code>
        /// </summary>
        public string BuildAgentPrompt()
        {
            // Collect tool-specific rules separately so we don't mutate Rules on
            // every call to BuildAgentPrompt() (the original code appended to Rules
            // directly, causing duplicates on repeated calls).
            var toolRules = ToolDefinitions.Count == 0
                ? []
                : new List<string>
                {
                    "- Respond ONLY in valid JSON. No prose. No markdown fences.",
                    "- Never invent tool names or argument values.",
                    """- If a required argument is missing, respond with: {"status":"error","message":"Missing required argument: <name>"}""",
                    "- Call one tool at a time. Wait for the result before proceeding.",
                    "- You only act using the tools provided."
                };

            var sb = new StringBuilder();

            // Role
            sb.AppendLine("## Role");
            sb.AppendLine($"You are {AgentName}, a {TemperaturePersonality(SamplingConfig.Temperature)} AI agent. " +
                          $"When asked for your name, you must respond with \"{AgentName}\".");

            // Rules
            var allRules = Rules.Concat(toolRules).ToList();
            if (allRules.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("## Rules");
                foreach (var rule in allRules)
                    sb.AppendLine(rule);
            }

            // Behaviors
            if (Behaviors.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("## Behavior");
                foreach (var behavior in Behaviors)
                    sb.AppendLine(behavior);
            }

            // Tools
            if (ToolDefinitions.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("## Tool Call Format");
                sb.AppendLine("""Respond ONLY with this JSON: { "tool": "<name>", "arguments": { ... } }""");

                sb.AppendLine();
                sb.AppendLine("## Available Tools");
                sb.AppendLine(ToolDefinitions.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

                sb.AppendLine();
                sb.AppendLine("## Final Response Format");
                sb.AppendLine("""{ "status": "success" | "error", "data": "<result>" }""");
            }

            // Skills
            if (Skills.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("## Skills");
                foreach (var skill in Skills)
                {
                    sb.AppendLine(skill);
                    sb.AppendLine(); // blank line between skill blocks
                }
            }

            // Delegated agents
            if (_agentsEnabled && _agents.Count > 0)
            {
                sb.AppendLine();
                sb.AppendLine("## Delegated Agents");
                sb.AppendLine("You may delegate sub-tasks to other agents using this format:");
                sb.AppendLine("""{{agent:<name>[:temp=<temperature>][:seed=<seed>]:<query>}}""");
                sb.AppendLine();
                sb.AppendLine("Available agents:");
                foreach (var agent in _agents.Values)
                {
                    var temp = agent.Config.Temperature.HasValue
                        ? $" (temp={agent.Config.Temperature.Value:F2})"
                        : "";
                    sb.AppendLine($"- {agent.Name}{temp}: {agent.Config.SystemPrompt[..Math.Min(agent.Config.SystemPrompt.Length, 120)]}");
                }
                sb.AppendLine();
                sb.AppendLine("After delegating, you will receive the result as a tool result. Continue from there.");
            }

            return sb.ToString().TrimEnd();
        }
    }
}
