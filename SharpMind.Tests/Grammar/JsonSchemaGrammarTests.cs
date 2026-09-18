using System.Text;
using SharpMind.Inference.Grammar;
using Xunit;

namespace SharpMind.Tests.Grammar;

/// <summary>
/// Milestone-4 tests for <see cref="JsonSchemaGrammar"/>: schema text is
/// translated to a GBNF grammar, then the resulting constraint is driven one
/// byte at a time. Because the pushdown engine is exact, a byte that cannot
/// continue the schema is masked immediately. These tests are what pinned the
/// NFA leak that made the earlier engine accept malformed structures.
/// </summary>
public sealed class JsonSchemaGrammarTests
{
    private static int Id(byte b) => TestTokenizer.IdFor(b);
    private static float[] Fresh() => new float[TestTokenizer.Count];

    private static IGrammarConstraint Constrain(string gbnf, params int[] stops)
        => GbnfGrammar.Parse(gbnf).CreateConstraint(
            TestTokenizer.ByteTable,
            stops.Length == 0 ? [TestTokenizer.EosId] : stops);

    /// <summary>Feeds <paramref name="json"/> byte-by-byte, asserting completeness at the end.</summary>
    private static bool Accepts(string gbnf, string json)
    {
        IGrammarConstraint c = Constrain(gbnf);
        foreach (byte b in Encoding.UTF8.GetBytes(json))
        {
            c.Apply(Fresh());
            int id = Id(b);
            if (!c.IsAllowed(id))
                return false;
            c.Accept(id);
        }
        c.Apply(Fresh());
        return c.CanStop;
    }

    private static bool SchemaAccepts(string schema, string json)
        => Accepts(JsonSchemaGrammar.FromSchema(schema), json);

    // ─── json_object ───────────────────────────────────────────────────────

    [Fact]
    public void AnyJson_AcceptsNestedValues()
    {
        Assert.True(Accepts(JsonSchemaGrammar.AnyJson, """
            {"a":1,"b":[2,3],"c":{"d":"x"},"e":true,"f":null,"g":-1.5e3}
            """));
        Assert.True(Accepts(JsonSchemaGrammar.AnyJson, "[[[[]]]]"));
        Assert.True(Accepts(JsonSchemaGrammar.AnyJson, "\"plain\""));
    }

    [Fact]
    public void AnyJson_RejectsMalformedStructures()
    {
        // "b":[2,3], -- a value is required after a comma.
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "{\"a\":}"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "{\"a\":1,}"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "[1,]"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "{,}"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "{\"a\":},"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, "{\"a\" 1}"));
        Assert.False(Accepts(JsonSchemaGrammar.AnyJson, ""));
    }

    // ─── Root objects ──────────────────────────────────────────────────────

    private const string PersonSchema = """
        {
          "type": "object",
          "properties": {
            "name":   { "type": "string" },
            "age":    { "type": "integer" },
            "active": { "type": "boolean" }
          }
        }
        """;

    [Fact]
    public void ObjectSchema_AcceptsValidAndRejectsViolations()
    {
        Assert.True(SchemaAccepts(PersonSchema, """{"name":"ada","age":36,"active":true}"""));
        Assert.True(SchemaAccepts(PersonSchema, """{"name":"","age":0,"active":false}"""));

        // Every declared property is required, in schema order.
        Assert.False(SchemaAccepts(PersonSchema, """{"name":"ada","active":true}""")); // missing age
        Assert.False(SchemaAccepts(PersonSchema, """{"age":36,"name":"ada","active":true}""")); // wrong order
        Assert.False(SchemaAccepts(PersonSchema, """{"name":"ada","age":"old","active":true}""")); // wrong type
        Assert.False(SchemaAccepts(PersonSchema, """{"name":"ada","age":36,"active":true,"extra":1}""")); // extra prop
        Assert.False(SchemaAccepts(PersonSchema, """{"name":,"age":36,"active":true}""")); // missing value
    }

    [Fact]
    public void ObjectSchema_WithoutExplicitType_UsesProperties()
    {
        Assert.True(SchemaAccepts("""{"properties":{"a":{"type":"string"}}}""", """{"a":"x"}"""));
        Assert.False(SchemaAccepts("""{"properties":{"a":{"type":"string"}}}""", """{"b":"x"}"""));
    }

    [Fact]
    public void NestedObject_RequiresEveryDeclaredProperty()
    {
        const string schema = """
            {
              "type": "object",
              "properties": {
                "user": {
                  "type": "object",
                  "properties": {
                    "id":   { "type": "integer" },
                    "tags": { "type": "array", "items": { "type": "string" } }
                  }
                }
              }
            }
            """;

        Assert.True(SchemaAccepts(schema, """{"user":{"id":7,"tags":["a","b"]}}"""));
        Assert.True(SchemaAccepts(schema, """{"user":{"id":7,"tags":[]}}"""));
        Assert.False(SchemaAccepts(schema, """{"user":{"id":7}}""")); // tags required
        Assert.False(SchemaAccepts(schema, """{"user":{"tags":["a"]}}""")); // id required
        Assert.False(SchemaAccepts(schema, """{"user":{"id":7,"tags":[1]}}""")); // wrong item type
    }

    // ─── Scalars ───────────────────────────────────────────────────────────

    [Fact]
    public void PrimitiveType_AcceptsOnlyThatType()
    {
        Assert.True(SchemaAccepts("""{"type":"string"}""", "\"hi\""));
        Assert.False(SchemaAccepts("""{"type":"string"}""", "\"hi"));
        Assert.False(SchemaAccepts("""{"type":"string"}""", "7"));

        Assert.True(SchemaAccepts("""{"type":"integer"}""", "-42"));
        Assert.True(SchemaAccepts("""{"type":"integer"}""", "0"));
        Assert.False(SchemaAccepts("""{"type":"integer"}""", "1.5"));
        Assert.False(SchemaAccepts("""{"type":"integer"}""", "\"1\""));

        Assert.True(SchemaAccepts("""{"type":"number"}""", "-1.5e3"));
        Assert.False(SchemaAccepts("""{"type":"number"}""", "1.5e"));

        Assert.True(SchemaAccepts("""{"type":"boolean"}""", "true"));
        Assert.True(SchemaAccepts("""{"type":"boolean"}""", "false"));
        Assert.False(SchemaAccepts("""{"type":"boolean"}""", "True"));

        Assert.True(SchemaAccepts("""{"type":"null"}""", "null"));
        Assert.False(SchemaAccepts("""{"type":"null"}""", "nil"));
    }

    [Fact]
    public void TypeArray_AllowsAnyMember()
    {
        const string schema = """{"type":["string","integer"]}""";
        Assert.True(SchemaAccepts(schema, "\"x\""));
        Assert.True(SchemaAccepts(schema, "7"));
        Assert.False(SchemaAccepts(schema, "true"));
        Assert.False(SchemaAccepts(schema, "null"));
    }

    // ─── enum / const ──────────────────────────────────────────────────────

    [Fact]
    public void Enum_AcceptsMembersOnly()
    {
        const string schema = """{"enum":["red","green","blue"]}""";
        Assert.True(SchemaAccepts(schema, "\"red\""));
        Assert.True(SchemaAccepts(schema, "\"green\""));
        Assert.False(SchemaAccepts(schema, "\"yellow\""));
        Assert.True(SchemaAccepts(schema, "\"blue\""));
    }

    [Fact]
    public void Enum_OfNumbers_MatchesRawValues()
    {
        const string schema = """{"enum":[1, 2.5]}""";
        Assert.True(SchemaAccepts(schema, "1"));
        Assert.True(SchemaAccepts(schema, "2.5"));
        Assert.False(SchemaAccepts(schema, "3"));
    }

    [Fact]
    public void Const_AcceptsExactValue()
    {
        Assert.True(SchemaAccepts("""{"const":42}""", "42"));
        Assert.False(SchemaAccepts("""{"const":42}""", "43"));
        Assert.False(SchemaAccepts("""{"const":42}""", "\"42\""));
    }

    // ─── Arrays ────────────────────────────────────────────────────────────

    [Fact]
    public void ArrayItems_EnforceItemType()
    {
        const string schema = """{"type":"array","items":{"type":"integer"}}""";
        Assert.True(SchemaAccepts(schema, "[1,2,3]"));
        Assert.True(SchemaAccepts(schema, "[]"));
        Assert.False(SchemaAccepts(schema, "[1,\"x\"]"));
        Assert.False(SchemaAccepts(schema, "[1,]"));
    }

    [Fact]
    public void UnspecifiedSchema_MatchesAnyJson()
    {
        Assert.True(SchemaAccepts("""{}""", """{"k":[1,{"nested":true}]}"""));
        Assert.True(SchemaAccepts("""{}""", "\"str\""));
        Assert.True(SchemaAccepts("{}", "-0.5"));
    }

    // ─── Rejection of unsupported schemas ──────────────────────────────────

    [Theory]
    [InlineData("""{"$ref":"#/definitions/x"}""")]
    [InlineData("""{"anyOf":[{"type":"string"}]}""")]
    [InlineData("""{"allOf":[{"type":"string"}]}""")]
    [InlineData("""{"oneOf":[{"type":"string"}]}""")]
    [InlineData("""{"type":"string","pattern":"^[a-z]+$"}""")]
    [InlineData("""{"type":"string","format":"email"}""")]
    [InlineData("""{"type":"integer","minimum":0}""")]
    [InlineData("""{"type":"string","minLength":2}""")]
    [InlineData("""{"type":"array","prefixItems":[{"type":"string"}]}""")]
    [InlineData("""{"type":"object","patternProperties":{"^x":{"type":"string"}}}""")]
    public void UnsupportedKeyword_Throws(string schema)
        => Assert.Throws<JsonSchemaException>(() => JsonSchemaGrammar.FromSchema(schema));

    [Fact]
    public void NonBooleanAdditionalProperties_Throws()
        => Assert.Throws<JsonSchemaException>(
            () => JsonSchemaGrammar.FromSchema("""{"type":"object","additionalProperties":{"type":"string"}}"""));

    [Fact]
    public void FromSchemaString_And_Element_Agree()
    {
        string gbnf = JsonSchemaGrammar.FromSchema(PersonSchema);
        Assert.StartsWith("root ::= ", gbnf);
        Assert.True(Accepts(gbnf, """{"name":"ada","age":36,"active":true}"""));
    }
}