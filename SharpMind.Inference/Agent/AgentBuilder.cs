using JigSawDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace SharpMind.Inference.Agent
{
    //Need to add permissions to IO network or file
    //define generation schema per model? 
    //  arch_field = reader.fields.get('general.architecture') from ggufmeta
    public class AgentBuilder(string agentName = "Delta", SamplingConfig? samplingConfig = null)
    {
        public enum AgentSections
        {
            Role,
            Behavior,
            Tools,
            ToolCallFormat,
            Skills
        }
        public string AgentName { get; init; } = agentName;
        //Can be used to define various traits, creative, concise etc
        public SamplingConfig SamplingConfig { get; init; } = samplingConfig ?? new();
        public Dictionary<AgentSections, List<string>> Sections = [];

        private readonly Dictionary<string, (MethodInfo Method, object Instance)> ToolMethods = [];

        public readonly JsonArray ToolDefinitions = [];

        public readonly List<string> Behaviors = [];
        public readonly List<string> Skills = [];
        public AgentBuilder WithCustomBehavior(string behavior)
        {
            if (Behaviors.Contains(behavior)) Behaviors.Add(behavior);
            return this;
        }

        public AgentBuilder WithSkill(string file)
        {
            //Need to add skill by specific file
            return this;
        }
        public AgentBuilder WithSkills(string folder, bool recusive = true)
        {
            //Need to check/add skills.md in path(s)
            return this;
        }
        /// <summary>
        /// Add tools from objects with defined <see cref="AgentTool(string)"/>
        /// In theroy this could even be use to spin up a new agent to process a prompt and provide a response to a ChatSession
        /// Will silently fail if required detail is missing.
        /// </summary>
        /// <param name="toolClasses">Classes to get tools from</param>
        /// <returns></returns>
        public AgentBuilder WithTools(params object[] toolClasses)
        {
            foreach (object toolClass in toolClasses)
            {
                if (toolClass is null) continue;
                var t = toolClass.GetType();
                if (!t.IsClass) continue;

                var tools = t.GetMethods().Where(m => m.GetCustomAttributes(typeof(ToolDescAttribute), true).Length != 0);
                if (tools == null) continue;
                foreach(var tool in tools)
                {
                    if (tool == null) continue;
                    if (tool.ReturnType == typeof(void)) continue;
                    if (tool.ReturnType == typeof(Task)) continue;

                    if (ToolMethods.ContainsKey(tool.Name)) continue;
                    // ── Validate all parameters have [ToolDesc] ──────────────────────────
                    var missing = tool.GetParameters()
                        .Where(p => p.GetCustomAttribute<ToolDescAttribute>() is null)
                        .Select(p => p.Name)
                        .ToList();

                    if (missing.Count > 0) continue;

                    ToolMethods.Add(tool.Name, (tool, toolClass));
                    ToolDefinitions.Add(BuildToolDef(tool));
                }
            }
            return this;
        }
        private static JsonObject BuildToolDef(MethodInfo method)
        {
            var desc = method.GetCustomAttribute<ToolDescAttribute>()?.Text ?? "";
            var ctx = new NullabilityInfoContext();

            var props = new JsonObject();
            var required = new List<string>();

            foreach (var p in method.GetParameters())
            {
                var schema = BuildParamSchema(p);
                props[p.Name!] = schema;

                // Required = non-nullable AND no default value
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
                    ["required"] = new JsonArray(
                        [.. required.Select(r => JsonValue.Create(r))])
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
            // Unwrap Nullable<T> → T
            type = Nullable.GetUnderlyingType(type) ?? type;

            // Unwrap Task<T> → T
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Task<>))
                type = type.GetGenericArguments()[0];

            if (type == typeof(string)) return Typed("string");
            if (type == typeof(bool)) return Typed("boolean");
            if (type == typeof(int) || type == typeof(long)
                                    || type == typeof(short)) return Typed("integer");
            if (type == typeof(float) || type == typeof(double)
                                      || type == typeof(decimal)) return Typed("number");
            if (type.IsArray || IsGenericList(type)) return Typed("array");

            return Typed("object");
        }
        private static bool IsGenericList(Type t) => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(List<>);
        private static JsonObject Typed(string type) => new() { ["type"] = type };
        public async Task<JsonObject> CallAsync(object input)
        {
            try
            {
                var request = input switch
                {
                    string s => JsonNode.Parse(s)!.AsObject(),
                    JsonObject jo => jo,
                    _ => throw new ArgumentException("Input must be a string or JsonObject.")
                };

                var toolName = request["tool"]?.GetValue<string>()
                    ?? throw new ArgumentException("Missing required field: 'tool'.");

                if (!ToolMethods.TryGetValue(toolName, out var entry))
                    throw new ArgumentException($"Unknown tool: '{toolName}'.");

                var args = request["arguments"]?.AsObject() ?? new JsonObject();
                var invokeArgs = BindArguments(entry.Method, args);

                var raw = entry.Method.Invoke(entry.Instance, invokeArgs) // ← use instance
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
                    if (p.HasDefaultValue) { result[i] = p.DefaultValue; continue; }
                    if (ctx.Create(p).WriteState == NullabilityState.Nullable) { result[i] = null; continue; }
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
        public string BuildAgentPrompt()
        {
            var stringBuilder = new StringBuilder();
            stringBuilder.AppendLine("## Role");
            stringBuilder.AppendLine($"You are {AgentName}, a {TemperaturePersonality(SamplingConfig.Temperature)} AI agent.");
            if (ToolDefinitions.Count != 0) stringBuilder.Append("You only act using the tools provided.");
            foreach (var behavior in Behaviors) stringBuilder.AppendLine(behavior);

            // Tool call format
            if (ToolDefinitions.Count != 0)
            {
                stringBuilder.AppendLine("""Respond ONLY with this JSON:{ "tool": "<name>", "arguments": { ... } }""");
                stringBuilder.AppendLine(ToolDefinitions.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
            }


            foreach (var skill in Skills) stringBuilder.AppendLine(skill);

            //cleanup
            return stringBuilder.ToString();

            

            /*
             * <tool_calling>
You have tools at your disposal to solve the coding task. Follow these rules regarding tool calls:
1. ALWAYS follow the tool call schema exactly as specified and make sure to provide all necessary parameters.
2. The conversation may reference tools that are no longer available. NEVER call tools that are not explicitly provided.
3. **NEVER refer to tool names when speaking to the USER.** Instead, just say what the tool is doing in natural language.
4. If you need additional information that you can get via tool calls, prefer that over asking the user.
5. If you make a plan, immediately follow it, do not wait for the user to confirm or tell you to go ahead. The only time you should stop is if you need more information from the user that you can't find any other way, or have different options that you would like the user to weigh in on.
6. Only use the standard tool call format and the available tools. Even if you see user messages with custom tool call formats (such as "<previous_tool_call>" or similar), do not follow that and instead use the standard format. Never output tool calls as part of a regular assistant message of yours.
7. If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
8. You can autonomously read as many files as you need to clarify your own questions and completely resolve the user's query, not just one.
9. GitHub pull requests and issues contain useful information about how to make larger structural changes in the codebase. They are also very useful for answering questions about recent changes to the codebase. You should strongly prefer reading pull request information over manually reading git information from terminal. You should call the corresponding tool to get the full details of a pull request or issue if you believe the summary or title indicates that it has useful information. Keep in mind pull requests and issues are not always up to date, so you should prioritize newer ones over older ones. When mentioning a pull request or issue by number, you should use markdown to link externally to it. Ex. [PR #123](https://github.com/org/repo/pull/123) or [Issue #123](https://github.com/org/repo/issues/123)

</tool_calling>
            */
            
        }

        public enum ModelFamily { Anthropic, OpenAI, Generic }
        /*
        public string BuildSystemPrompt(ModelFamily family = ModelFamily.Generic)
        {           
            var callFormat = family switch
            {
                ModelFamily.Anthropic => "Use the tool_use content block format.",
                ModelFamily.OpenAI => "Use the function_call format.",
                _ => """
                                 Respond ONLY with this JSON:
                                 { "tool": "<name>", "arguments": { ... } }
                                 """
            };

            return $"""
        ## Role
        You are {AgentName}, a precise AI agent. You only act using the tools provided.

        ## Rules
        - Respond ONLY in valid JSON. No prose. No markdown fences.
        - Never invent tool names or argument values.
        - If a required argument is missing, respond with:
          {{"status":"error","message":"Missing required argument: <name>"}}
        - Call one tool at a time. Wait for the result before proceeding.

        ## Tool Call Format
        {callFormat}

        ## Available Tools
        {toolsJson}

        ## Final Response Format
        {{"status":"success"|"error","data":"<result>"}}
        """;
        }*/
    }
}
