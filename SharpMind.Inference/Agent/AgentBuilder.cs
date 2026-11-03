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
                // Classified by the marker interfaces the tool classes implement, not by
                // their type names. A tool that is both file and network is File.
                var declaring = entry.Method.DeclaringType;
                if (declaring is not null && typeof(SharpMind.Core.AgentTools.IFileToolService).IsAssignableFrom(declaring))
                    return ToolCategory.File;
                if (declaring is not null && typeof(SharpMind.Core.AgentTools.INetworkToolService).IsAssignableFrom(declaring))
                    return ToolCategory.Network;
                return ToolCategory.General;
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
                    if (TryFindConflictingTool(tool.Name, out var existing))
                    {
                        // Overloads cannot be told apart by name, so the second one is refused and
                        // reported, instead of being dropped without a trace. Names that differ only
                        // by case or separators are refused for the same reason: a model cannot
                        // distinguish them and recovery must never pick one at random.
                        RegistrationWarnings.Add(string.Equals(existing, tool.Name, StringComparison.Ordinal)
                            ? $"Tool '{tool.Name}' is declared more than once; only the first overload is registered."
                            : $"Tool '{tool.Name}' conflicts with registered tool '{existing}' (names differ only by case or separators); only '{existing}' is registered.");
                        continue;
                    }

                    // All parameters must carry [ToolDesc]
                    var missingAnnotations = tool.GetParameters()
                        .Where(p => !IsRuntimeParameter(p) && p.GetCustomAttribute<ToolDescAttribute>() is null)
                        .Select(p => p.Name)
                        .ToList();

                    if (missingAnnotations.Count > 0) continue;

                    var duplicateParameters = tool.GetParameters()
                        .Where(p => !IsRuntimeParameter(p))
                        .Select(p => p.Name!)
                        .GroupBy(ToolIdentifier.Normalize)
                        .FirstOrDefault(group => group.Count() > 1);
                    if (duplicateParameters is not null)
                    {
                        RegistrationWarnings.Add(
                            $"Tool '{tool.Name}' declares parameters that differ only by case or separators ({string.Join(", ", duplicateParameters)}); argument binding is ambiguous.");
                    }

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
            if (TryFindConflictingTool(name, out var existing))
            {
                if (!string.Equals(existing, name, StringComparison.Ordinal))
                {
                    RegistrationWarnings.Add(
                        $"Tool '{name}' conflicts with registered tool '{existing}' (names differ only by case or separators); only '{existing}' is registered.");
                }
                return this;
            }
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

        /// <summary>
        /// Finds an already-registered tool whose name is the same, differs only
        /// by case, or differs only by separators. Such names cannot be told
        /// apart reliably by a model or by recovery, so the second registration
        /// is refused and reported.
        /// </summary>
        private bool TryFindConflictingTool(string name, out string? existing)
        {
            var normalized = ToolIdentifier.Normalize(name);
            foreach (var candidate in ToolMethods.Keys.Concat(_delegateTools.Keys))
            {
                if (string.Equals(candidate, name, StringComparison.OrdinalIgnoreCase)
                    || ToolIdentifier.Normalize(candidate) == normalized)
                {
                    existing = candidate;
                    return true;
                }
            }

            existing = null;
            return false;
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
                // Supplied by the runtime, not the model: never part of the schema.
                if (IsRuntimeParameter(p)) continue;

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

        /// <summary>
        /// A parameter the runtime supplies: a cancellation token. It is excluded from the
        /// tool schema and from the required-argument check, and bound to the call's token.
        /// </summary>
        internal static bool IsRuntimeParameter(ParameterInfo p) =>
            p.ParameterType == typeof(CancellationToken);

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
                var required = ToolCallExample.RequiredNames(tool);
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

        /// <summary>
        /// The tool the prompt examples show: the first real (non-demo) tool that has
        /// required arguments, falling back to the first real tool.
        /// </summary>
        private JsonObject? ExampleTool()
        {
            var tools = ToolDefinitions.OfType<JsonObject>().ToList();
            var real = tools.Where(t => !IsDemoTool(t)).ToList();
            if (real.Count == 0) real = tools;
            return real.FirstOrDefault(t => ToolCallExample.RequiredNames(t).Count > 0) ?? real.FirstOrDefault();
        }

        /// <summary>The example call for the active format, filled in with realistic values.</summary>
        private string BuildExampleCall(JsonObject tool)
        {
            string name = tool["name"]?.GetValue<string>() ?? "tool";
            return ToolCallExample.Build(CallFormat, name, ToolCallExample.ExampleArguments(tool));
        }

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
        public async Task<JsonObject> CallToolAsync(JsonObject toolCall, CancellationToken cancellationToken = default)
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
                {
                    return Error(
                        $"Unknown tool: '{toolName}'. Valid tools: {string.Join(", ", RegisteredToolNames)}.",
                        "unknown_tool",
                        ClosestToolName(toolName));
                }

                var reflectionArgs = toolCall["arguments"]?.AsObject() ?? [];
                var invokeArgs = BindArguments(entry.Method, reflectionArgs, out var ignored, cancellationToken);

                var raw = entry.Method.Invoke(entry.Instance, invokeArgs)
                    ?? throw new InvalidOperationException("Tool returned null.");

                var response = Success(ResultText(await ResolveResultAsync(raw)));
                if (ignored.Count > 0)
                    response["notes"] = $"Ignored argument(s) the tool does not declare: {string.Join(", ", ignored)}.";
                return response;
            }
            catch (ToolBindingException ex) { return Error(ex.Message, "invalid_arguments"); }
            catch (TargetInvocationException ex) { return Error(ex.InnerException?.Message ?? ex.Message); }
            catch (Exception ex) { return Error(ex.Message); }
        }

        /// <summary>
        /// The registered tool whose name is the unique closest match to an
        /// unknown name, for a retry suggestion only. Never used to dispatch.
        /// </summary>
        private string? ClosestToolName(string requested)
        {
            var normalized = ToolIdentifier.Normalize(requested);
            if (normalized.Length < 4) return null;

            var scored = RegisteredToolNames
                .Select(name =>
                {
                    var candidate = ToolIdentifier.Normalize(name);
                    return (Name: name, Distance: ToolIdentifier.Distance(normalized, candidate));
                })
                .Where(x => ToolIdentifier.IsWithinFuzzyThreshold(
                    normalized, ToolIdentifier.Normalize(x.Name), x.Distance))
                .OrderBy(x => x.Distance)
                .ToList();

            if (scored.Count == 0) return null;
            int best = scored[0].Distance;
            var winners = scored.Where(x => x.Distance == best).Select(x => x.Name).ToList();
            return winners.Count == 1 ? winners[0] : null;
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
        internal static object?[] BindArguments(MethodInfo method, JsonObject args, out List<string> ignored,
            CancellationToken cancellationToken = default)
        {
            var ctx = new NullabilityInfoContext();
            var @params = method.GetParameters();
            var result = new object?[@params.Length];
            var declaredNames = @params.Where(p => !IsRuntimeParameter(p)).Select(p => p.Name!).ToList();
            ignored = args.Select(kv => kv.Key).Where(k => !IsBoundKey(k, declaredNames)).ToList();

            for (int i = 0; i < @params.Length; i++)
            {
                var p = @params[i];
                if (IsRuntimeParameter(p))
                {
                    result[i] = cancellationToken;
                    continue;
                }
                var node = FindArgument(args, p.Name!);

                if (node is null)
                {
                    if (p.HasDefaultValue)
                    { result[i] = p.DefaultValue; continue; }
                    if (ctx.Create(p).WriteState == NullabilityState.Nullable)
                    { result[i] = null; continue; }
                    throw new ToolBindingException(
                        $"Required argument '{p.Name}' is missing. Parameters: {DescribeParameters(method)}.");
                }

                try
                {
                    result[i] = CoerceValue(node, p.ParameterType);
                }
                catch (Exception ex) when (ex is ArgumentException or JsonException or FormatException
                                              or OverflowException or InvalidCastException)
                {
                    throw new ToolBindingException(
                        $"Argument '{p.Name}' expects {ExpectedType(p.ParameterType)} but got {Describe(node)}. " +
                        $"Parameters: {DescribeParameters(method)}.");
                }
            }

            return result;
        }

        /// <summary>
        /// True when a supplied key binds to a declared parameter: exact,
        /// case-insensitive, or a unique separator-insensitive match.
        /// </summary>
        private static bool IsBoundKey(string key, IReadOnlyList<string> declaredNames)
        {
            if (declaredNames.Any(d => string.Equals(d, key, StringComparison.OrdinalIgnoreCase)))
                return true;

            var normalized = ToolIdentifier.Normalize(key);
            return declaredNames.Count(d => ToolIdentifier.Normalize(d) == normalized) == 1;
        }

        /// <summary>
        /// The argument with this name: exact, then case-insensitive, then
        /// separator-insensitive (<c>action_type</c> matches <c>actionType</c>).
        /// A normalized match is used only when exactly one supplied key maps to
        /// it, so duplicate names never bind silently. Null-valued arguments
        /// count as absent.
        /// </summary>
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

            if (node is null)
            {
                var normalized = ToolIdentifier.Normalize(name);
                string? matchedKey = null;
                bool ambiguous = false;
                foreach (var kv in args)
                {
                    if (ToolIdentifier.Normalize(kv.Key) != normalized) continue;
                    if (matchedKey is not null)
                    {
                        ambiguous = true;
                        break;
                    }
                    matchedKey = kv.Key;
                }
                if (!ambiguous && matchedKey is not null)
                    node = args[matchedKey];
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

        private static JsonObject Error(string message, string? code = null, string? suggestedTool = null)
        {
            var error = new JsonObject { ["status"] = "error", ["message"] = message };
            if (code is not null) error["code"] = code;
            if (suggestedTool is not null) error["suggestedTool"] = suggestedTool;
            return error;
        }
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
        /// Tool-calling rules for the active format. These are the rule sets that worked for
        /// the small models: each format has its own wording for how to emit a call, and
        /// they are kept as they were. Changing them changes model behaviour.
        /// </summary>
        private List<string> BuildToolRules() => CallFormat switch
        {
            ToolCallFormat.Qwen =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual/conversational questions directly—only call a tool in ## Available Tools for explicit actions, UI interactions, or capabilities that require it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "If using a tool, call one tool at a time, wait for its result, then answer directly from it—never answer before the result arrives. For list parameters pass a JSON array in the argument; unknown args: say so briefly.",
                "To call a tool, reply with ONLY the JSON object and nothing else, then stop—no preamble, no follow-up text, no restating the arguments.",
                "If the user's request does not match a tool in ## Available Tools, answer entirely in plain prose: never emit a JSON object, never name or reference tools, and behave exactly as if tools did not exist."
            ],
            ToolCallFormat.QwenXml =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual and conversational questions directly; only call a tool for an explicit action or a capability that needs it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "Make at most one tool call per reply. Wait for its result, then answer directly from it; never answer before the result arrives. Pass each argument as its own parameter; list values are JSON arrays. Unknown arguments: say so briefly.",
                "To call a tool, reply with ONLY the one <tool_call> block shown under ## Tool Call Format, then stop: no preamble, no follow-up text, no restating the arguments. Use only the tool names from your function definitions; never invent names or values, and never write tool names, descriptions or schemas as text.",
                "If the user's request does not match one of your tools, answer entirely in plain prose: never emit a <tool_call> block, never name or reference tools, and behave exactly as if tools did not exist."
            ],
            ToolCallFormat.Mistral =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual/conversational questions directly—only call a tool in ## Available Tools for explicit actions, UI interactions, or capabilities that require it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "If using a tool, call one tool at a time, wait for its result, then answer directly from it—never answer before the result arrives. For list parameters pass a JSON array in the argument; unknown args: say so briefly.",
                "To call a tool, reply with ONLY [TOOL_CALLS] followed by exactly one JSON object {\"name\":\"...\",\"arguments\":{...}} and nothing else, then stop—no preamble, no follow-up text, no restating the arguments. Use only tools in ## Available Tools; never invent tool names or values.",
                "If the user's request does not match a tool in ## Available Tools, answer entirely in plain prose: never emit a JSON object or [TOOL_CALLS], never name or reference tools, and behave exactly as if tools did not exist."
            ],
            ToolCallFormat.Llama3 =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual/conversational questions directly—only call a tool in ## Available Tools for explicit actions, UI interactions, or capabilities that require it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "If using a tool, call one tool at a time, wait for its result, then answer directly from it—never answer before the result arrives. For list parameters pass a JSON array in the argument; unknown args: say so briefly.",
                "To call a tool, reply with ONLY exactly one JSON object {\"name\":\"...\",\"arguments\":{...}} and nothing else, then stop—no preamble, no follow-up text, no restating the arguments. Use only tools in ## Available Tools; never invent tool names or values.",
                "If the user's request does not match a tool in ## Available Tools, answer entirely in plain prose: never emit a JSON object, never name or reference tools, and behave exactly as if tools did not exist."
            ],
            ToolCallFormat.Gemma =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual/conversational questions directly—only call a tool in ## Available Tools for explicit actions, UI interactions, or capabilities that require it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "If using a tool, call one tool at a time, wait for its result, then answer directly from it—never answer before the result arrives. For list parameters pass a JSON array in the args field; unknown args: say so briefly.",
                "To call a tool, reply with ONLY exactly one JSON object {\"name\":\"...\",\"args\":{...}} using \"args\" and not \"arguments\", and nothing else, then stop—no preamble, no follow-up text, no restating the arguments. Use only tools in ## Available Tools; never invent tool names or values.",
                "If the user's request does not match a tool in ## Available Tools, answer entirely in plain prose: never emit a JSON object, never name or reference tools, and behave exactly as if tools did not exist."
            ],
            _ =>
            [
                "You are an agent with the ability to call tools, only call tools if it is best suited to the response. Answer factual/conversational questions directly—only call a tool in ## Available Tools for explicit actions, UI interactions, or capabilities that require it. Treat \"can you...\" and \"could you please...\" as orders, not questions.",
                "If using a tool, call one tool at a time, wait for its result, then answer directly from it—never answer before the result arrives. Narrate freely, then place one tool call as exactly one JSON object inside \u003Ctool_call\u003E tags; everything outside the tags is narration—write it in character. For list parameters pass a JSON array in the argument; unknown args: say so briefly. Use only provided tools; never invent names or values.",
                "To call a tool, place exactly one JSON object inside \u003Ctool_call\u003E tags and then stop; never emit raw JSON outside the tags and no follow-up text after the call—no restating the arguments.",
                "If the user's request does not match a tool in ## Available Tools, answer entirely in plain prose: never emit a JSON object, never use \u003Ctool_call\u003E tags, never name or reference tools, and behave exactly as if tools did not exist."
            ]
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
                var example = ExampleTool();
                switch (CallFormat)
                {
                    case ToolCallFormat.Qwen:
                        sb.AppendLine("Only when the request is an action performed by one of the tools in ## Available Tools, reply with exactly one JSON object — nothing else. Example:");
                        break;
                    case ToolCallFormat.QwenXml:
                        sb.AppendLine("Only when the request is an action performed by one of your tools, reply with exactly one tool call block - nothing else. Example:");
                        break;
                    case ToolCallFormat.Mistral:
                        sb.AppendLine("Call a tool by replying with [TOOL_CALLS] followed by exactly one JSON object. Example:");
                        break;
                    case ToolCallFormat.Llama3:
                        sb.AppendLine("Call a tool by replying with exactly one JSON object — nothing else. Example:");
                        break;
                    case ToolCallFormat.Gemma:
                        sb.AppendLine("Call a tool by replying with exactly one JSON object using \"args\" (not \"arguments\"). Example:");
                        break;
                    default:
                        sb.AppendLine("Narrate in prose; to act, place exactly one JSON object inside these tags:");
                        break;
                }
                if (example is not null)
                    sb.AppendLine(BuildExampleCall(example));
                sb.AppendLine();
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
