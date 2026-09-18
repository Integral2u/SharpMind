using System.Text;
using System.Text.Json;

namespace SharpMind.Inference.Grammar;

/// <summary>
/// Builds a GBNF grammar from a JSON Schema, so constrained decoding can force
/// a model to emit a schema-shaped JSON document.
/// </summary>
/// <remarks>
/// Supported keywords: <c>type</c> (a string or array of strings, including
/// <c>null</c>), <c>properties</c>, <c>items</c>, <c>enum</c>, <c>const</c> and
/// <c>additionalProperties</c> (boolean). Objects emit every declared property
/// in schema order with no extras, which satisfies both optional-property
/// schemas and OpenAI's strict <c>additionalProperties:false</c> requirement.
/// Anything outside this subset (for example <c>$ref</c>, <c>oneOf</c>,
/// <c>pattern</c>) throws <see cref="JsonSchemaException"/> rather than silently
/// producing unconstrained output.
/// </remarks>
public static class JsonSchemaGrammar
{
    /// <summary>GBNF matching any single JSON value (used for <c>json_object</c>).</summary>
    public static string AnyJson { get; } = new Builder().BuildAny();

    /// <summary>Build a GBNF grammar matching an instance of <paramref name="schema"/>.</summary>
    public static string FromSchema(JsonElement schema)
    {
        var builder = new Builder();
        string root = builder.Build(schema);
        return builder.Emit(root);
    }

    /// <summary>Build a GBNF grammar from raw schema JSON.</summary>
    public static string FromSchema(string schemaJson)
    {
        using var doc = JsonDocument.Parse(schemaJson);
        return FromSchema(doc.RootElement.Clone());
    }

    private sealed class Builder
    {
        private const string Ws = "jws";
        private const string Str = "jstr";
        private const string StrChar = "jstrchar";
        private const string StrEsc = "jstrchar_esc";
        private const string Int = "jint";
        private const string Num = "jnum";
        private const string Any = "jval";
        private const string Obj = "jobj";
        private const string Arr = "jarr";
        private const string Member = "jmember";

        private readonly List<string> _defs = [];
        private bool _sharedAdded;
        private int _next;

        public string Build(JsonElement schema)
        {
            if (schema.ValueKind != JsonValueKind.Object)
                throw new JsonSchemaException("a JSON schema must be an object");

            RejectUnsupported(schema);

            if (schema.TryGetProperty("enum", out var enumProp) && enumProp.ValueKind == JsonValueKind.Array)
                return BuildEnum(enumProp);

            if (schema.TryGetProperty("const", out var constProp))
                return Define(Literal(constProp.GetRawText()));

            if (schema.TryGetProperty("type", out var typeProp))
                return BuildTyped(typeProp, schema);

            if (schema.TryGetProperty("properties", out _))
                return BuildObject(schema);

            if (schema.TryGetProperty("items", out _))
                return BuildArray(schema);

            EnsureShared();
            return Any;
        }

        public string BuildAny()
        {
            EnsureShared();
            return Emit(Any);
        }

        public string Emit(string root)
        {
            EnsureShared();
            var sb = new StringBuilder();
            sb.Append("root ::= ").Append(root).Append('\n');
            foreach (string def in _defs)
                sb.Append(def).Append('\n');
            return sb.ToString();
        }

        private string BuildEnum(JsonElement values)
        {
            var parts = new List<string>();
            foreach (var value in values.EnumerateArray())
                parts.Add(Literal(value.GetRawText()));
            if (parts.Count == 0)
                throw new JsonSchemaException("'enum' must contain at least one value");
            return Define(string.Join(" | ", parts));
        }

        private string BuildTyped(JsonElement typeProp, JsonElement schema)
        {
            if (typeProp.ValueKind == JsonValueKind.String)
                return BuildSingle(typeProp.GetString()!, schema);

            if (typeProp.ValueKind == JsonValueKind.Array)
            {
                var parts = new List<string>();
                foreach (var t in typeProp.EnumerateArray())
                {
                    if (t.ValueKind != JsonValueKind.String)
                        throw new JsonSchemaException("'type' array entries must be strings");
                    parts.Add(BuildSingle(t.GetString()!, schema));
                }
                if (parts.Count == 0)
                    throw new JsonSchemaException("'type' must not be an empty array");
                return parts.Count == 1 ? parts[0] : Define(string.Join(" | ", parts));
            }

            throw new JsonSchemaException("'type' must be a string or an array of strings");
        }

        private string BuildSingle(string type, JsonElement schema) => type switch
        {
            "object" => BuildObject(schema),
            "array" => BuildArray(schema),
            "string" => Shared(Str),
            "integer" => Shared(Int),
            "number" => Shared(Num),
            "boolean" => Define($"{Literal("true")} | {Literal("false")}"),
            "null" => Define(Literal("null")),
            _ => throw new JsonSchemaException($"unsupported JSON schema type '{type}'"),
        };

        private string BuildObject(JsonElement schema)
        {
            EnsureShared();
            if (!schema.TryGetProperty("properties", out var props) || props.ValueKind != JsonValueKind.Object)
                return Obj;

            var members = new List<string>();
            foreach (var prop in props.EnumerateObject())
            {
                string valueRule = Build(prop.Value);
                // The key literal must include the surrounding quotes ("name").
                string keyLiteral = Literal("\"" + prop.Name + "\"");
                members.Add($"{keyLiteral} {Ws} {Literal(":")} {Ws} {valueRule}");
            }

            if (members.Count == 0)
                return Obj;

            string joined = string.Join($" {Ws} {Literal(",")} {Ws} ", members);
            return Define($"{Literal("{")} {Ws} ( {joined} ) {Ws} {Literal("}")}");
        }

        private string BuildArray(JsonElement schema)
        {
            EnsureShared();
            if (!schema.TryGetProperty("items", out var items) || items.ValueKind != JsonValueKind.Object)
                return Arr;

            string itemRule = Build(items);
            return Define($"{Literal("[")} {Ws} ( {itemRule} ( {Ws} {Literal(",")} {Ws} {itemRule} )* )? {Ws} {Literal("]")}");
        }

        private static readonly string[] UnsupportedKeywords =
        [
            "$ref", "$defs", "definitions", "oneOf", "anyOf", "allOf", "not",
            "pattern", "format", "minimum", "maximum", "exclusiveMinimum",
            "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
            "minItems", "maxItems", "minProperties", "maxProperties",
            "patternProperties", "propertyNames", "prefixItems", "contains",
            "uniqueItems", "if", "then", "else", "dependentSchemas",
        ];

        private static void RejectUnsupported(JsonElement schema)
        {
            foreach (string keyword in UnsupportedKeywords)
            {
                if (schema.TryGetProperty(keyword, out _))
                    throw new JsonSchemaException($"unsupported JSON schema keyword '{keyword}'");
            }

            if (schema.TryGetProperty("additionalProperties", out var extra)
                && extra.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            {
                throw new JsonSchemaException("'additionalProperties' must be a boolean");
            }
        }

        private string Define(string expression)
        {
            string name = "j" + _next++;
            _defs.Add($"{name} ::= {expression}");
            return name;
        }

        private string Shared(string name)
        {
            EnsureShared();
            return name;
        }

        private void EnsureShared()
        {
            if (_sharedAdded)
                return;
            _sharedAdded = true;

            _defs.Add($"{Ws} ::= [ \\t\\n\\r]*");
            _defs.Add($"{StrEsc} ::= [\"\\\\/bfnrt] | {Literal("u")} [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]");
            _defs.Add($"{StrChar} ::= [^\"\\\\\\x00-\\x1F] | {Literal("\\")} {StrEsc}");
            _defs.Add($"{Str} ::= {Literal("\"")} {StrChar}* {Literal("\"")}");
            _defs.Add($"{Int} ::= {Literal("-")}? ( {Literal("0")} | [1-9] [0-9]* )");
            _defs.Add($"{Num} ::= {Literal("-")}? ( {Literal("0")} | [1-9] [0-9]* ) ( {Literal(".")} [0-9]+ )? ( [eE] [+-]? [0-9]+ )?");
            _defs.Add($"{Any} ::= {Obj} | {Arr} | {Str} | {Num} | {Literal("true")} | {Literal("false")} | {Literal("null")}");
            _defs.Add($"{Member} ::= {Str} {Ws} {Literal(":")} {Ws} {Any}");
            _defs.Add($"{Obj} ::= {Literal("{")} {Ws} ( {Member} ( {Ws} {Literal(",")} {Ws} {Member} )* )? {Ws} {Literal("}")}");
            _defs.Add($"{Arr} ::= {Literal("[")} {Ws} ( {Any} ( {Ws} {Literal(",")} {Ws} {Any} )* )? {Ws} {Literal("]")}");
        }

        /// <summary>Wrap <paramref name="text"/> as a GBNF string literal, escaping it.</summary>
        private static string Literal(string text)
        {
            var sb = new StringBuilder(text.Length + 2);
            sb.Append('"');
            foreach (char c in text)
            {
                switch (c)
                {
                    case '\\':
                        sb.Append("\\\\");
                        break;
                    case '"':
                        sb.Append("\\\"");
                        break;
                    default:
                        if (c < 0x20)
                            sb.Append("\\x").Append(((int)c).ToString("X2"));
                        else
                            sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }
    }
}

/// <summary>Thrown when a JSON schema uses constructs the GBNF builder cannot represent.</summary>
public sealed class JsonSchemaException : Exception
{
    public JsonSchemaException(string message) : base(message) { }
}
