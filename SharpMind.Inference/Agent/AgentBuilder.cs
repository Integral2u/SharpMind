using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using SharpMind.Core;
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
        public SamplingConfig SamplingConfig { get; set; } = samplingConfig ?? new();
        public ToolCallFormat CallFormat { get; set; } = ToolCallFormat.SharpMind;
        public IContextCompactor? Compactor { get; set; }
        public IReadOnlyList<IContextCompactor> PluginCompactors { get; set; } = [];
        public IReadOnlyList<IPromptPreProcessor> PluginPreProcessors { get; set; } = [];
        public IReadOnlyList<IPromptPostProcessor> PluginPostProcessors { get; set; } = [];
        public HashSet<string> DisabledTools { get; set; } = [];

        public IReadOnlyList<string> RegisteredToolNames => [.. ToolMethods.Keys, .. _delegateTools.Keys];

        /// <summary>Problems found while registering tools (for example overloads), for the host to show or log.</summary>
        public List<string> RegistrationWarnings { get; } = [];

        /// <summary>
        /// Best-effort IO category for a registered tool. Reflection-registered
        /// built-ins are classified by their declaring type; delegate tools fall
        /// back to their name. Used to pre-flight permission checks for tools
        /// (such as the built-in file system tool) that do not use the
        /// intercepting IO abstractions themselves.
        /// </summary>
        /// <summary>
        /// True for a reflection-registered tool whose code lives outside the
        /// built-in assemblies (SharpMind.Core, SharpMind.CUI). Such code can
        /// do IO the interceptors never see, so the session must gate it.
        /// </summary>
        public bool IsExternalTool(string toolName)
        {
            if (!ToolMethods.TryGetValue(toolName, out var entry)) return false;
            string? assembly = entry.Method.DeclaringType?.Assembly.GetName().Name;
            return assembly is not ("SharpMind.Core" or "SharpMind.CUI");
        }

        public ToolCategory GetToolCategory(string toolName)
        {
            if (ToolMethods.TryGetValue(toolName, out var entry))
            {
                return entry.Method.DeclaringType?.Name switch
                {
                    "FileSystemTool" or "GitTool" or "GrepTool" => ToolCategory.File,
                    "WeatherTool" => ToolCategory.Network,
                    _ => ToolCategory.General
                };
            }

            // Delegate tools own their IO; they are gated by the intercepting
            // filesystem/network abstractions instead, not by this pre-flight.
            return ToolCategory.General;
        }

        // Not currently used in prompt building but available for callers that want to
        // inspect or manipulate sections as keyed lists.
        public Dictionary<AgentSections, List<string>> Sections = [];
        private readonly Dictionary<string, (MethodInfo Method, object Instance)> ToolMethods = [];
        private readonly Dictionary<string, (string Name, string Description, JsonObject Schema, Func<JsonObject, Task<string>> Execute)> _delegateTools = [];
        public JsonArray ToolDefinitions { get; } = [];

        public readonly List<string> Behaviors = [];
        public readonly List<string> Skills = [];
        public readonly List<string> Rules = [];
        private readonly List<string> _additionalSystemPrompts = [];
        public IReadOnlyList<string> AdditionalSystemPrompts => _additionalSystemPrompts;

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
        /// Appends ready-made skill content (markdown) directly to the Skills
        /// section, without requiring a physical file. Used for skills embedded
        /// inside an .SMM container.
        /// </summary>
        public IAgentBuilder WithSkillContent(string content)
        {
            if (string.IsNullOrWhiteSpace(content))
                return this;

            content = content.Trim();
            if (content.Length == 0)
                return this;

            if (!Skills.Contains(content))
                Skills.Add(content);

            return this;
        }

        /// <summary>
        /// Adds a standalone system message that is inserted at the top of the
        /// history, before the synthesized agent prompt. Used for the default
        /// system prompt embedded inside an .SMM container.
        /// </summary>
        public IAgentBuilder WithAdditionalSystemPrompt(string prompt)
        {
            if (string.IsNullOrWhiteSpace(prompt))
                return this;

            prompt = prompt.Trim();
            if (prompt.Length == 0)
                return this;

            if (!_additionalSystemPrompts.Contains(prompt))
                _additionalSystemPrompts.Add(prompt);

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
                    if (DisabledTools.Contains(tool.Name)) continue;
                    if (ToolMethods.ContainsKey(tool.Name))
                    {
                        // Overloads cannot be told apart by name, so the second one is refused and
                        // reported, instead of being dropped without a trace.
                        RegistrationWarnings.Add($"Tool '{tool.Name}' is declared more than once; only the first overload is registered.");
                        continue;
                    }

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

        /// <summary>
        /// Registers a delegate-based tool with an explicit name, description,
        /// and JSON Schema — no reflection needed. Designed for external callers
        /// (e.g. the Microsoft.Extensions.AI adapter) that supply their own
        /// tool metadata and invocation logic.
        /// </summary>
        public IAgentBuilder WithTool(string name, string description, JsonObject schema, Func<JsonObject, Task<string>> execute)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(name);
            ArgumentNullException.ThrowIfNull(schema);
            ArgumentNullException.ThrowIfNull(execute);
            if (ToolMethods.ContainsKey(name) || _delegateTools.ContainsKey(name))
                return this;
            if (DisabledTools.Contains(name))
                return this;

            _delegateTools[name] = (name, description, schema, execute);
            ToolDefinitions.Add(new JsonObject
            {
                ["name"] = name,
                ["description"] = description,
                ["parameters"] = schema
            });
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

            var range = param.GetCustomAttribute<ToolRangeAttribute>();
            if (range is not null)
            {
                if (range.HasMinimum) schema["minimum"] = JsonValue.Create(range.Minimum);
                if (range.HasMaximum) schema["maximum"] = JsonValue.Create(range.Maximum);
            }

            if (param.HasDefaultValue && param.DefaultValue is not null)
                schema["default"] = DefaultNode(param.DefaultValue);

            return schema;
        }

        /// <summary>A parameter default as a typed JSON value, not its ToString.</summary>
        private static JsonNode DefaultNode(object value)
        {
            if (value.GetType().IsEnum)
                return JsonValue.Create(Enum.GetName(value.GetType(), value) ?? value.ToString())!;
            return value switch
            {
                string s => JsonValue.Create(s)!,
                bool b => JsonValue.Create(b)!,
                int i => JsonValue.Create(i)!,
                long l => JsonValue.Create(l)!,
                short sh => JsonValue.Create((int)sh)!,
                float f => JsonValue.Create(f)!,
                double d => JsonValue.Create(d)!,
                decimal m => JsonValue.Create(m)!,
                _ => JsonValue.Create(value.ToString())!
            };
        }

        private static JsonObject JsonTypeToSchema(Type type)
        {
            type = Nullable.GetUnderlyingType(type) ?? type;

            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Task<>))
                type = type.GetGenericArguments()[0];

            if (type == typeof(string)) return Typed("string");
            if (type == typeof(bool)) return Typed("boolean");
            if (type.IsEnum)
            {
                var names = Enum.GetNames(type).Select(n => JsonValue.Create(n)!).ToList();
                return new JsonObject { ["type"] = "string", ["enum"] = new JsonArray([.. names]) };
            }
            if (type == typeof(int) || type == typeof(long) || type == typeof(short)) return Typed("integer");
            if (type == typeof(float) || type == typeof(double) || type == typeof(decimal)) return Typed("number");
            if (type.IsArray) return ArraySchema(type.GetElementType()!);
            if (IsGenericList(type)) return ArraySchema(type.GetGenericArguments()[0]);

            return Typed("object");
        }

        private static JsonObject ArraySchema(Type element)
        {
            var schema = Typed("array");
            schema["items"] = JsonTypeToSchema(element);
            return schema;
        }

        private static bool IsGenericList(Type t) =>
            t.IsGenericType && t.GetGenericTypeDefinition() == typeof(List<>);
        private static JsonObject Typed(string type) => new() { ["type"] = type };

        /// <summary>
        /// Budget for the tool list in the system prompt, in estimated tokens. Over
        /// budget, the list is re-rendered in shorter tiers rather than cut mid-JSON.
        /// The CUI re-prefills the whole conversation each turn, so a smaller system
        /// prompt is seconds saved on a slow engine.
        /// </summary>
        public int ToolListTokenBudget { get; set; } = 1000;

        /// <summary>
        /// When false, the system prompt lists tool names only, because the active
        /// chat template already renders the full definitions. The session sets this
        /// from its formatter; it stays true for formatters that do not render tools.
        /// </summary>
        public bool ToolListInPrompt { get; set; } = true;

        /// <summary>Rough token estimate for prompt budgeting: about four characters per token.</summary>
        internal static int EstimateTokens(string text) => (text.Length + 3) / 4;

        /// <summary>
        /// The tool list for the system prompt. Full definitions when they fit the
        /// budget; otherwise progressively shorter tiers, and names only as the last
        /// resort. Every tier is valid JSON, and anything cut is reported after it.
        /// </summary>
        private string BuildCompactToolList()
        {
            string argKey = CallFormat == ToolCallFormat.Gemma ? "args" : "arguments";
            string full = ToolDefinitions.ToJsonString();
            if (EstimateTokens(full) <= ToolListTokenBudget)
                return full;

            for (int tier = 1; tier <= 3; tier++)
            {
                var list = new JsonArray();
                foreach (var item in ToolDefinitions.OfType<JsonObject>())
                    list.Add(CompactTool(item, tier, argKey));
                string json = list.ToJsonString();
                if (EstimateTokens(json) <= ToolListTokenBudget)
                    return json + "\n(Descriptions were shortened to fit the prompt.)";
            }

            // Last resort: as many tool names as fit, with the rest reported.
            var names = ToolDefinitions.OfType<JsonObject>()
                .Select(t => t["name"]?.GetValue<string>() ?? "?")
                .ToList();
            var kept = new List<string>();
            foreach (var name in names)
            {
                if (EstimateTokens(JsonSerializer.Serialize(kept.Append(name))) > ToolListTokenBudget) break;
                kept.Add(name);
            }
            string nameJson = JsonSerializer.Serialize(kept);
            return kept.Count < names.Count
                ? nameJson + $"\n({names.Count - kept.Count} more tool(s) not listed: {string.Join(", ", names.Skip(kept.Count))})"
                : nameJson;
        }

        /// <summary>
        /// One tool at a compaction tier. Tier 1 keeps a short description and the
        /// descriptions of required arguments; tier 2 shortens both; tier 3 keeps only
        /// names, argument names and types. Required names are always kept.
        /// </summary>
        private static JsonObject CompactTool(JsonObject tool, int tier, string argKey)
        {
            var copy = new JsonObject { ["name"] = JsonValue.Create(tool["name"]?.GetValue<string>() ?? "?") };
            int maxDesc = tier switch { 1 => 120, 2 => 60, _ => 0 };
            string desc = tool["description"]?.GetValue<string>() ?? "";
            if (maxDesc > 0 && desc.Length > 0)
                copy["description"] = JsonValue.Create(Clip(FirstSentence(desc), maxDesc));

            if (tool["parameters"] is JsonObject pars && pars["properties"] is JsonObject props)
            {
                var required = RequiredNames(tool);
                var args = new JsonObject();
                foreach (var (name, schema) in props)
                {
                    var s = schema as JsonObject;
                    string type = s?["type"]?.GetValue<string>() ?? "string";
                    string? argDesc = s?["description"]?.GetValue<string>();
                    if (tier == 1 && required.Contains(name) && argDesc is not null)
                        args[name] = new JsonObject { ["type"] = type, ["description"] = Clip(argDesc, 80) };
                    else
                        args[name] = JsonValue.Create(type);
                }
                copy[argKey] = args;
                if (required.Count > 0)
                    copy["required"] = new JsonArray(required.Select(n => (JsonNode)JsonValue.Create(n)!).ToArray());
            }
            return copy;
        }

        private static string FirstSentence(string text)
        {
            int dot = text.IndexOf(". ", StringComparison.Ordinal);
            return dot > 0 ? text[..(dot + 1)] : text;
        }

        private static string Clip(string text, int max) =>
            text.Length <= max ? text : text[..max].TrimEnd() + "…";

        private static bool IsDemoTool(JsonObject tool) =>
            (tool["name"]?.GetValue<string>() ?? "").StartsWith("Echo", StringComparison.Ordinal);

        private static List<string> RequiredNames(JsonObject tool)
        {
            var names = new List<string>();
            if (tool["parameters"] is JsonObject pars && pars["required"] is JsonArray req)
                foreach (var r in req)
                    if (r?.GetValue<string>() is { } n)
                        names.Add(n);
            return names;
        }

        /// <summary>
        /// The tool the prompt examples show: the first real (non-demo) tool that has
        /// required arguments, falling back to the first real tool.
        /// </summary>
        private JsonObject? ExampleTool()
        {
            var tools = ToolDefinitions.OfType<JsonObject>().ToList();
            var real = tools.Where(t => !IsDemoTool(t)).ToList();
            if (real.Count == 0) real = tools;
            return real.FirstOrDefault(t => RequiredNames(t).Count > 0) ?? real.FirstOrDefault();
        }

        /// <summary>The example call for the active format, filled in with realistic values.</summary>
        private string BuildExampleCall(JsonObject tool)
        {
            string name = tool["name"]?.GetValue<string>() ?? "tool";
            var args = ExampleArguments(tool);

            if (CallFormat == ToolCallFormat.QwenXml)
            {
                var sb = new StringBuilder();
                sb.AppendLine("<tool_call>");
                sb.AppendLine($"<function={name}>");
                foreach (var (key, value) in args)
                {
                    sb.AppendLine($"<parameter={key}>");
                    sb.AppendLine(ValueText(value));
                    sb.AppendLine("</parameter>");
                }
                sb.Append("</function>\n</tool_call>");
                return sb.ToString();
            }

            string argKey = CallFormat == ToolCallFormat.Gemma ? "args" : "arguments";
            string json = new JsonObject { ["name"] = name, [argKey] = args }.ToJsonString();
            return CallFormat switch
            {
                ToolCallFormat.Mistral => "[TOOL_CALLS]" + json,
                ToolCallFormat.SharpMind => "<tool_call>" + json + "</tool_call>",
                _ => json
            };
        }

        /// <summary>
        /// Arguments for an example call: every required argument (up to three) and
        /// at most one optional one, so the example stays short.
        /// </summary>
        private static JsonObject ExampleArguments(JsonObject tool)
        {
            var args = new JsonObject();
            if (tool["parameters"] is not JsonObject pars || pars["properties"] is not JsonObject props)
                return args;

            var required = RequiredNames(tool);
            var chosen = props.Where(p => required.Contains(p.Key)).Take(3)
                .Concat(props.Where(p => !required.Contains(p.Key)).Take(1))
                .ToList();
            foreach (var (name, schema) in chosen)
                args[name] = SampleValue(name, schema as JsonObject);
            return args;
        }

        private static JsonNode SampleValue(string name, JsonObject? schema)
        {
            if (schema?["enum"] is JsonArray en && en.Count > 0 && en[0] is not null)
                return en[0]!.DeepClone();

            string type = schema?["type"]?.GetValue<string>() ?? "string";
            switch (type)
            {
                case "integer":
                    return JsonValue.Create(3)!;
                case "number":
                    return JsonValue.Create(2.5)!;
                case "boolean":
                    return JsonValue.Create(true)!;
                case "array":
                    bool choices = name.Contains("option", StringComparison.OrdinalIgnoreCase)
                        || name.Contains("choice", StringComparison.OrdinalIgnoreCase);
                    return choices ? new JsonArray("red", "green", "blue") : new JsonArray(SampleText(name));
                default:
                    return JsonValue.Create(SampleText(name))!;
            }
        }

        private static string SampleText(string argName)
        {
            string n = argName.ToLowerInvariant();
            if (n.Contains("city") || n.Contains("location") || n.Contains("place")) return "Paris";
            if (n.Contains("path") || n.Contains("file")) return "notes.txt";
            if (n.Contains("dir") || n.Contains("folder")) return ".";
            if (n.Contains("url")) return "https://example.com";
            if (n.Contains("query") || n.Contains("pattern") || n.Contains("search")) return "meeting";
            if (n.Contains("content") || n.Contains("text") || n.Contains("message")) return "Hello";
            if (n.EndsWith("name", StringComparison.Ordinal)) return "Alex";
            return "example";
        }

        private static string ValueText(JsonNode? value) =>
            value is JsonValue jv && jv.TryGetValue<string>(out var s) ? s : value?.ToJsonString() ?? "";

        /// <summary>Sample result text for the prompt's result example.</summary>
        private static string ExampleResultText(string toolName) =>
            toolName.Contains("weather", StringComparison.OrdinalIgnoreCase)
                ? "21C, sunny"
                : "3 matches: red, green, blue";

        /// <summary>Sample reply, written from the sample result above.</summary>
        private static string ExampleAnswer(string toolName) =>
            toolName.Contains("weather", StringComparison.OrdinalIgnoreCase)
                ? "It is 21C and sunny in Paris."
                : "The three matches are red, green and blue.";
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
        /// The one-line purpose the model sees for a sub-agent: its configured
        /// <see cref="AgentConfig.Purpose"/>, or the start of its system prompt.
        /// </summary>
        internal static string AgentPurpose(IAgent agent)
        {
            string text = agent.Config.Purpose ?? agent.Config.SystemPrompt;
            string line = text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "";
            return line.Length > 120 ? line[..120] + "..." : line;
        }

        /// <summary>
        /// Exposes delegation as an ordinary tool: <c>delegate(agent, task)</c>, with the
        /// agent names as an enum and a one-line purpose for each. Registered once, after
        /// agents are created, and only when there is at least one agent.
        /// </summary>
        public void RegisterDelegateTool()
        {
            if (_agents.Count == 0 || _delegateTools.ContainsKey("delegate")) return;

            var names = _agents.Keys.Select(n => (JsonNode)JsonValue.Create(n)!).ToList();
            var purposes = _agents.Values.Select(a => $"{a.Name}: {AgentPurpose(a)}");
            var schema = new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject
                {
                    ["agent"] = new JsonObject
                    {
                        ["type"] = "string",
                        ["enum"] = new JsonArray([.. names]),
                        ["description"] = string.Join("; ", purposes)
                    },
                    ["task"] = new JsonObject
                    {
                        ["type"] = "string",
                        ["description"] = "The complete sub-task, with all the context the agent needs."
                    }
                },
                ["required"] = new JsonArray("agent", "task")
            };
            WithTool("delegate",
                "Hands a self-contained sub-task to a specialist agent and returns its answer. Use it when a listed agent's purpose matches the task better than you can do it yourself.",
                schema,
                _ => Task.FromResult(""));   // dispatched by ChatSession, which owns the sub-agent run
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

                // Delegate-based tool (added via WithTool(name, desc, schema, fn))
                if (_delegateTools.TryGetValue(toolName, out var delegateEntry))
                {
                    var delegateArgs = toolCall["arguments"]?.AsObject() ?? [];
                    var result = await delegateEntry.Execute(delegateArgs);
                    return Success(result);
                }

                // Reflection-based tool (added via WithTools(typeof(MyTools)))
                if (!ToolMethods.TryGetValue(toolName, out var entry))
                    throw new ArgumentException(
                        $"Unknown tool: '{toolName}'. Valid tools: {string.Join(", ", RegisteredToolNames)}.");

                var reflectionArgs = toolCall["arguments"]?.AsObject() ?? [];
                var invokeArgs = BindArguments(entry.Method, reflectionArgs, out var ignored);

                var raw = entry.Method.Invoke(entry.Instance, invokeArgs)
                    ?? throw new InvalidOperationException("Tool returned null.");

                var response = Success(ResultText(await ResolveResultAsync(raw)));
                if (ignored.Count > 0)
                    response["notes"] = $"Ignored argument(s) the tool does not declare: {string.Join(", ", ignored)}.";
                return response;
            }
            catch (TargetInvocationException ex) { return Error(ex.InnerException?.Message ?? ex.Message); }
            catch (Exception ex) { return Error(ex.Message); }
        }

        /// <summary>
        /// Awaits a tool's return value when it is a Task or ValueTask, with or without
        /// a result, and yields the result. Any other value is returned unchanged.
        /// </summary>
        internal static async Task<object?> ResolveResultAsync(object raw)
        {
            var type = raw.GetType();
            Task? task = raw as Task;
            bool isValueTask = type == typeof(ValueTask)
                || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTask<>));
            if (task is null && isValueTask)
                task = (Task?)type.GetMethod("AsTask")!.Invoke(raw, null);
            if (task is null)
                return raw;

            await task.ConfigureAwait(false);
            return task.GetType().GetProperty("Result")?.GetValue(task);
        }

        /// <summary>Tool output as text. JSON nodes keep their JSON; numbers use invariant culture.</summary>
        internal static string ResultText(object? value) => value switch
        {
            null => "",
            string s => s,
            JsonNode n => n.ToJsonString(),
            IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? ""
        };

        /// <summary>
        /// Binds the model's arguments to the method's parameters. Names match
        /// case-insensitively and values are coerced leniently (see <see cref="CoerceValue"/>).
        /// Arguments the method does not declare are returned in <paramref name="ignored"/>.
        /// A value that cannot be coerced fails with the expected type and the tool's parameters.
        /// </summary>
        internal static object?[] BindArguments(MethodInfo method, JsonObject args, out List<string> ignored)
        {
            var ctx = new NullabilityInfoContext();
            var @params = method.GetParameters();
            var result = new object?[@params.Length];
            var declared = new HashSet<string>(@params.Select(p => p.Name!), StringComparer.OrdinalIgnoreCase);
            ignored = args.Select(kv => kv.Key).Where(k => !declared.Contains(k)).ToList();

            for (int i = 0; i < @params.Length; i++)
            {
                var p = @params[i];
                var node = FindArgument(args, p.Name!);

                if (node is null)
                {
                    if (p.HasDefaultValue)
                    { result[i] = p.DefaultValue; continue; }
                    if (ctx.Create(p).WriteState == NullabilityState.Nullable)
                    { result[i] = null; continue; }
                    throw new ArgumentException(
                        $"Required argument '{p.Name}' is missing. Parameters: {DescribeParameters(method)}.");
                }

                try
                {
                    result[i] = CoerceValue(node, p.ParameterType);
                }
                catch (Exception ex) when (ex is ArgumentException or JsonException or FormatException
                                              or OverflowException or InvalidCastException)
                {
                    throw new ArgumentException(
                        $"Argument '{p.Name}' expects {ExpectedType(p.ParameterType)} but got {Describe(node)}. " +
                        $"Parameters: {DescribeParameters(method)}.");
                }
            }

            return result;
        }

        /// <summary>The argument with this name, matched case-insensitively. Null-valued arguments count as absent.</summary>
        private static JsonNode? FindArgument(JsonObject args, string name)
        {
            JsonNode? node = args[name];
            if (node is null)
            {
                foreach (var kv in args)
                {
                    if (string.Equals(kv.Key, name, StringComparison.OrdinalIgnoreCase))
                    {
                        node = kv.Value;
                        break;
                    }
                }
            }
            return node is not null && node.GetValueKind() == JsonValueKind.Null ? null : node;
        }

        /// <summary>
        /// Converts a JSON value to a parameter type, leniently: numeric and boolean text
        /// parse ("5" to 5, "true" to true), scalars become text for string parameters
        /// (123 to "123"), a single value is accepted for a list, and enum names match
        /// case-insensitively. Anything else throws ArgumentException.
        /// </summary>
        internal static object? CoerceValue(JsonNode node, Type targetType)
        {
            targetType = Nullable.GetUnderlyingType(targetType) ?? targetType;

            if (targetType == typeof(string)) return ScalarText(node);
            if (targetType == typeof(bool)) return ToBool(node);
            if (targetType.IsEnum) return ToEnum(node, targetType);
            if (targetType == typeof(int) || targetType == typeof(long) || targetType == typeof(short)
                || targetType == typeof(float) || targetType == typeof(double) || targetType == typeof(decimal))
                return ToNumber(node, targetType);
            if (targetType.IsArray) return ToArray(node, targetType.GetElementType()!);
            if (IsGenericList(targetType)) return ToList(node, targetType.GetGenericArguments()[0]);

            return node.Deserialize(targetType);
        }

        private static string ScalarText(JsonNode node) => node.GetValueKind() switch
        {
            JsonValueKind.String => node.GetValue<string>(),
            _ => node.ToJsonString()   // numbers, booleans, and JSON objects/arrays keep their text
        };

        private static bool ToBool(JsonNode node)
        {
            switch (node.GetValueKind())
            {
                case JsonValueKind.True: return true;
                case JsonValueKind.False: return false;
                case JsonValueKind.String when bool.TryParse(node.GetValue<string>().Trim(), out bool b):
                    return b;
            }
            throw new ArgumentException("is not true or false");
        }

        private static object ToEnum(JsonNode node, Type enumType)
        {
            if (node.GetValueKind() == JsonValueKind.String)
            {
                string text = node.GetValue<string>().Trim();
                foreach (string name in Enum.GetNames(enumType))
                    if (string.Equals(name, text, StringComparison.OrdinalIgnoreCase))
                        return Enum.Parse(enumType, name);
            }
            throw new ArgumentException("is not a known value");
        }

        private static object ToNumber(JsonNode node, Type targetType)
        {
            string text = node.GetValueKind() switch
            {
                JsonValueKind.Number => node.ToJsonString(),
                JsonValueKind.String => node.GetValue<string>().Trim(),
                _ => throw new ArgumentException("is not a number")
            };

            if (targetType == typeof(decimal))
            {
                if (!decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal dec))
                    throw new ArgumentException("is not a number");
                return dec;
            }
            if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double d) || !double.IsFinite(d))
                throw new ArgumentException("is not a number");

            if (targetType == typeof(double)) return d;
            if (targetType == typeof(float)) return (float)d;
            if (d != Math.Truncate(d)) throw new ArgumentException("is not a whole number");
            if (targetType == typeof(short)) return checked((short)d);
            if (targetType == typeof(int)) return checked((int)d);
            return checked((long)d);
        }

        /// <summary>The elements of a list value. A string holding a JSON array is parsed; a lone value is a one-element list.</summary>
        private static IEnumerable<JsonNode> Elements(JsonNode node)
        {
            if (node is JsonArray arr)
                return arr.Where(n => n is not null).Select(n => n!);

            if (node.GetValueKind() == JsonValueKind.String && node.GetValue<string>().TrimStart().StartsWith('['))
            {
                try
                {
                    if (JsonNode.Parse(node.GetValue<string>()) is JsonArray parsed)
                        return parsed.Where(n => n is not null).Select(n => n!).ToList();
                }
                catch (JsonException) { }
            }
            return [node];
        }

        private static object ToArray(JsonNode node, Type element)
        {
            var items = Elements(node).Select(n => CoerceValue(n, element)).ToList();
            var array = Array.CreateInstance(element, items.Count);
            for (int i = 0; i < items.Count; i++)
                array.SetValue(items[i], i);
            return array;
        }

        private static object ToList(JsonNode node, Type element)
        {
            var list = (System.Collections.IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(element))!;
            foreach (var n in Elements(node))
                list.Add(CoerceValue(n, element));
            return list;
        }

        /// <summary>What a parameter accepts, in words for error messages.</summary>
        private static string ExpectedType(Type t)
        {
            var schema = JsonTypeToSchema(t);
            string type = schema["type"]?.GetValue<string>() ?? "value";
            if (schema["enum"] is JsonArray values)
                return "one of " + string.Join(", ", values.Select(v => v!.GetValue<string>()));
            if (type == "array" && schema["items"] is JsonObject items)
                return $"array of {items["type"]?.GetValue<string>() ?? "value"}";
            return type;
        }

        private static string DescribeParameters(MethodInfo method)
        {
            var ctx = new NullabilityInfoContext();
            return string.Join(", ", method.GetParameters().Select(p =>
            {
                bool required = !p.HasDefaultValue && ctx.Create(p).WriteState != NullabilityState.Nullable;
                return $"{p.Name} ({ExpectedType(p.ParameterType)}{(required ? ", required" : "")})";
            }));
        }

        private static string Describe(JsonNode node)
        {
            string text = node.GetValueKind() switch
            {
                JsonValueKind.String => "\"" + node.GetValue<string>() + "\"",
                JsonValueKind.Array => "an array",
                JsonValueKind.Object => "an object",
                _ => node.ToJsonString()
            };
            return text.Length > 60 ? text[..60] + "..." : text;
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
        /// Rules every tool-calling format shares, each stated once. Format-specific
        /// wording lives only in <see cref="ToolCallLine"/>.
        /// </summary>
        private static readonly string[] SharedToolRules =
        [
            "Use a tool only when the request needs an action or information that a tool in the Available Tools list provides. Otherwise answer directly. Treat \"can you...\" and \"could you...\" as requests to do the thing.",
            "Call at most one tool per reply. When you call a tool, the reply contains only the call; the result arrives next, and then you answer the user from it.",
            "If no listed tool fits the request, say so in one sentence and say what you can help with.",
            "If a required argument is missing or unclear, ask the user one short question instead of calling the tool.",
            "Copy tool and argument names exactly as listed in the Available Tools list, and never invent values."
        ];

        /// <summary>The shared tool-calling rules for the system prompt.</summary>
        private List<string> BuildToolRules() => [.. SharedToolRules];

        /// <summary>The one line that says how to emit a call in the active format.</summary>
        private string ToolCallLine() => CallFormat switch
        {
            ToolCallFormat.QwenXml =>
                "To call a tool, reply with exactly one block of this form: <tool_call><function=TOOL><parameter=ARG>VALUE</parameter></function></tool_call>.",
            ToolCallFormat.Mistral =>
                "To call a tool, reply with [TOOL_CALLS] followed by one JSON object: {\"name\":\"TOOL\",\"arguments\":{...}}.",
            ToolCallFormat.Llama3 or ToolCallFormat.Qwen =>
                "To call a tool, reply with one JSON object: {\"name\":\"TOOL\",\"arguments\":{...}}.",
            ToolCallFormat.Gemma =>
                "To call a tool, reply with one JSON object: {\"name\":\"TOOL\",\"args\":{...}}.",
            _ =>
                "To call a tool, reply with one block: <tool_call>{\"name\":\"TOOL\",\"arguments\":{...}}</tool_call>."
        };
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
            // Tool rules are built per call rather than appended to Rules, so repeated
            // calls never duplicate them. With CallFormat.None the whole tools section
            // is omitted, so an unknown model is never drifted towards a call shape.
            bool hasTools = ToolDefinitions.Count > 0 && CallFormat != ToolCallFormat.None;
            var toolRules = hasTools ? BuildToolRules() : [];

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

            // Tools: one call format, one filled-in example, one tool-free example.
            if (hasTools)
            {
                sb.AppendLine();
                sb.AppendLine("## Tool Call Format");
                sb.AppendLine(ToolCallLine());
                var example = ExampleTool();
                if (example is not null)
                {
                    string exampleName = example["name"]?.GetValue<string>() ?? "tool";
                    sb.AppendLine("Example call:");
                    sb.AppendLine(BuildExampleCall(example));
                    sb.AppendLine("The result arrives in this form, and you reply from it:");
                    sb.AppendLine($"[{exampleName} result] {ExampleResultText(exampleName)}");
                    sb.AppendLine($"Reply: {ExampleAnswer(exampleName)}");
                    sb.AppendLine();
                }
                sb.AppendLine("Example with no tool needed:");
                sb.AppendLine("User: hello");
                sb.AppendLine("Reply: Hi! How can I help you today?");

                sb.AppendLine();
                sb.AppendLine("## Available Tools");
                if (ToolListInPrompt)
                {
                    sb.AppendLine(BuildCompactToolList());
                }
                else
                {
                    var names = ToolDefinitions.OfType<JsonObject>().Select(t => t["name"]?.GetValue<string>() ?? "?");
                    sb.AppendLine("Names: " + string.Join(", ", names) + ". Their full definitions are provided separately.");
                }
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
                if (hasTools)
                {
                    // Delegation is the delegate tool (listed under Available Tools); the
                    // agents and their purposes are carried by its schema.
                    sb.AppendLine("To hand a self-contained sub-task to a specialist, call the delegate tool. Its agent argument lists the available agents.");
                }
                else
                {
                    sb.AppendLine("Available agents:");
                    foreach (var agent in _agents.Values)
                        sb.AppendLine($"- {agent.Name}: {AgentPurpose(agent)}");
                }
                sb.AppendLine();
                sb.AppendLine("After delegating, you will receive the result as a tool result. Continue from there.");
            }

            return sb.ToString().TrimEnd();
        }
    }
}
