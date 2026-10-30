using System.Reflection;
using System.Text.Json.Nodes;
using SharpMind.Core;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>Tool-side binding fixture: each method is a registered tool.</summary>
public sealed class BindingProbeTools
{
    public static int LastCount;
    public static string? LastLabel;

    [ToolDesc("Counts things.")]
    public string Count([ToolDesc("How many.")] int count, [ToolDesc("What.")] string label)
    {
        LastCount = count;
        LastLabel = label;
        return $"{label} x{count}";
    }

    [ToolDesc("Takes a flag.")]
    public string Flag([ToolDesc("On or off.")] bool enabled) => enabled ? "on" : "off";

    [ToolDesc("Takes a list.")]
    public string Pick([ToolDesc("Choices.")] List<string> options) => string.Join("|", options);

    [ToolDesc("Takes an array of numbers.")]
    public string Sum([ToolDesc("Values.")] int[] values) => values.Sum().ToString();

    [ToolDesc("Takes a level.")]
    public string Level([ToolDesc("Level.")] Level level) => level.ToString();

    [ToolDesc("Async result.")]
    public async Task<string> Async([ToolDesc("Word.")] string word)
    {
        await Task.Delay(1);
        return "async:" + word;
    }

    [ToolDesc("Value-task result.")]
    public ValueTask<int> ValueAsync([ToolDesc("Number.")] int n) => ValueTask.FromResult(n * 2);

    [ToolDesc("Default and range.")]
    public string Ranged([ToolDesc("Percent.")][ToolRange(Minimum = 0, Maximum = 100, HasMinimum = true, HasMaximum = true)] int percent = 50) => percent.ToString();
}

public enum Level { Low, High }

public sealed class ArgumentBindingTests
{
    private static MethodInfo M(string name) => typeof(BindingProbeTools).GetMethod(name)!;

    private static JsonObject Args(string json) => (JsonObject)JsonNode.Parse(json)!;

    private static IAgentBuilder Agent()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new BindingProbeTools());
        return agent;
    }

    private static async Task<JsonObject> Call(string tool, string argsJson)
    {
        var agent = Agent();
        return await agent.CallToolAsync(new JsonObject
        {
            ["tool"] = tool,
            ["arguments"] = Args(argsJson)
        });
    }

    // ── (c) coercion ───────────────────────────────────────────────

    [Fact]
    public void NumericString_CoercesToInt()
    {
        var bound = AgentBuilder.BindArguments(M("Count"), Args("""{"count":"5","label":"x"}"""), out _);
        Assert.Equal(5, bound[0]);
    }

    [Fact]
    public void Number_ToString_KeepsItsText()
    {
        var bound = AgentBuilder.BindArguments(M("Count"), Args("""{"count":3,"label":123}"""), out _);
        Assert.Equal(3, bound[0]);
        Assert.Equal("123", bound[1]);
    }

    [Theory]
    [InlineData("\"true\"", true)]
    [InlineData("\"False\"", false)]
    [InlineData("true", true)]
    public void BooleanText_Coerces(string json, bool expected)
    {
        var bound = AgentBuilder.BindArguments(M("Flag"), Args("{\"enabled\":" + json + "}"), out _);
        Assert.Equal(expected, bound[0]);
    }

    [Fact]
    public void ArgumentNames_MatchCaseInsensitively()
    {
        var bound = AgentBuilder.BindArguments(M("Count"), Args("""{"COUNT":"2","Label":"y"}"""), out var ignored);
        Assert.Equal(2, bound[0]);
        Assert.Equal("y", bound[1]);
        Assert.Empty(ignored);
    }

    [Fact]
    public void UnknownExtraArgument_IsIgnoredAndReported()
    {
        var bound = AgentBuilder.BindArguments(M("Count"), Args("""{"count":1,"label":"z","colour":"red"}"""), out var ignored);
        Assert.Equal(1, bound[0]);
        Assert.Equal(["colour"], ignored);
    }

    [Fact]
    public void JsonArrayText_ForAListParameter_IsParsed()
    {
        var bound = AgentBuilder.BindArguments(M("Pick"), Args("""{"options":"[\"red\",\"green\"]"}"""), out _);
        Assert.Equal(["red", "green"], (List<string>)bound[0]!);
    }

    [Fact]
    public void SingleValue_ForAListParameter_BecomesOneElement()
    {
        var bound = AgentBuilder.BindArguments(M("Pick"), Args("""{"options":"red"}"""), out _);
        Assert.Equal(["red"], (List<string>)bound[0]!);
    }

    [Fact]
    public void Array_OfIntegersCoercesEachElement()
    {
        var bound = AgentBuilder.BindArguments(M("Sum"), Args("""{"values":["1","2",3]}"""), out _);
        Assert.Equal(new[] { 1, 2, 3 }, (int[])bound[0]!);
    }

    [Fact]
    public void Enum_MatchesNameCaseInsensitively()
    {
        var bound = AgentBuilder.BindArguments(M("Level"), Args("""{"level":"high"}"""), out _);
        Assert.Equal(Level.High, bound[0]);
    }

    [Fact]
    public void FractionalNumber_ForIntParameter_IsRejectedWithTypeInError()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            AgentBuilder.BindArguments(M("Count"), Args("""{"count":"2.5","label":"x"}"""), out _));
        Assert.Contains("'count' expects integer but got \"2.5\"", ex.Message);
        Assert.Contains("count (integer, required)", ex.Message);
    }

    // ── (f)-style precise errors through the public dispatch ───────────

    [Fact]
    public async Task MissingRequiredArgument_ErrorListsParameters()
    {
        var result = await Call("Count", """{"label":"x"}""");
        Assert.Equal("error", result["status"]!.GetValue<string>());
        string message = result["message"]!.GetValue<string>();
        Assert.Contains("Required argument 'count' is missing", message);
        Assert.Contains("count (integer, required)", message);
        Assert.Contains("label (string, required)", message);
    }

    [Fact]
    public async Task UnknownToolName_ErrorListsValidNames()
    {
        var result = await Call("Nope", "{}");
        Assert.Equal("error", result["status"]!.GetValue<string>());
        string message = result["message"]!.GetValue<string>();
        Assert.Contains("Unknown tool: 'Nope'", message);
        Assert.Contains("Valid tools: Count", message);
    }

    // ── Schema (item 6) ────────────────────────────────────────────

    [Fact]
    public void ListParameter_SchemaHasItems()
    {
        var agent = Agent();
        var pick = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "Pick");
        var options = pick["parameters"]!["properties"]!["options"]!;
        Assert.Equal("array", options["type"]!.GetValue<string>());
        Assert.Equal("string", options["items"]!["type"]!.GetValue<string>());
    }

    [Fact]
    public void EnumParameter_SchemaHasEnumValues()
    {
        var agent = Agent();
        var level = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "Level");
        var values = level["parameters"]!["properties"]!["level"]!["enum"]!.AsArray();
        Assert.Equal(["Low", "High"], values.Select(v => v!.GetValue<string>()));
    }

    [Fact]
    public void Range_AndTypedDefault_AppearInSchema()
    {
        var agent = Agent();
        var ranged = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "Ranged");
        var percent = ranged["parameters"]!["properties"]!["percent"]!;
        Assert.Equal(0, percent["minimum"]!.GetValue<double>());
        Assert.Equal(100, percent["maximum"]!.GetValue<double>());
        Assert.Equal(50, percent["default"]!.GetValue<int>());   // a number, not the text "50"
    }

    // ── (b-part) result handling (item 8) ───────────────────────────

    [Fact]
    public async Task TaskOfString_ResultIsAwaited()
    {
        var result = await Call("Async", """{"word":"hi"}""");
        Assert.Equal("async:hi", result["data"]!.GetValue<string>());
    }

    [Fact]
    public async Task ValueTaskOfInt_ResultIsAwaited()
    {
        var result = await Call("ValueAsync", """{"n":4}""");
        Assert.Equal("8", result["data"]!.GetValue<string>());
    }

    [Fact]
    public async Task DefaultArgument_IsUsedWhenOmitted()
    {
        var result = await Call("Ranged", "{}");
        Assert.Equal("50", result["data"]!.GetValue<string>());
    }

    [Fact]
    public async Task ExtraArgument_IsReportedInResultNotes()
    {
        var result = await Call("Count", """{"count":1,"label":"z","colour":"red"}""");
        Assert.Equal("success", result["status"]!.GetValue<string>());
        Assert.Contains("colour", result["notes"]!.GetValue<string>());
    }
}
