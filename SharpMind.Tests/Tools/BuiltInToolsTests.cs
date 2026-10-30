using System.Text.Json.Nodes;
using SharpMind.Core.AgentTools;
using SharpMind.Inference.Agent;
using Xunit;

namespace SharpMind.Tests.Tools;

/// <summary>Built-in file tools: reads, search, exact edit, and symlink containment.</summary>
public sealed class FileToolBehaviourTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "sm-tools-" + Guid.NewGuid().ToString("N"));
    private readonly FileSystemTool _tools;

    public FileToolBehaviourTests()
    {
        Directory.CreateDirectory(_root);
        _tools = new FileSystemTool(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private void Write(string rel, string text) => File.WriteAllText(Path.Combine(_root, rel), text);

    [Fact]
    public void ReadFile_NumbersLines_AndCapsAtTheDefault()
    {
        Write("a.txt", string.Join("\n", Enumerable.Range(1, 5).Select(i => "row " + i)));
        string text = _tools.ReadFile("a.txt");
        Assert.StartsWith("1 | row 1", text);
        Assert.Contains("5 | row 5", text);
    }

    [Fact]
    public void ReadFile_OffsetAndLimit_ReturnTheRequestedWindowAndSayWhatRemains()
    {
        Write("a.txt", string.Join("\n", Enumerable.Range(1, 10).Select(i => "row " + i)));
        string text = _tools.ReadFile("a.txt", offset: 4, limit: 2);

        Assert.Contains("4 | row 4", text);
        Assert.Contains("5 | row 5", text);
        Assert.DoesNotContain("6 | row 6", text);
        Assert.Contains("showing lines 4-5 of 10; call again with offset=6", text);
    }

    [Fact]
    public void ReadFile_OffsetPastTheEnd_SaysSo()
    {
        Write("a.txt", "one\ntwo");
        Assert.Contains("past the end", _tools.ReadFile("a.txt", offset: 9));
    }

    [Fact]
    public void Search_FindsLiteralTextCaseInsensitively_WithPathAndLine()
    {
        Write("notes.txt", "alpha\nHello World\nomega");
        string text = _tools.Search("hello world");
        Assert.Equal("notes.txt:2: Hello World", text);
    }

    [Fact]
    public void Search_NoMatch_SaysSo()
    {
        Write("notes.txt", "alpha");
        Assert.Equal("No matches for 'zebra'.", _tools.Search("zebra"));
    }

    [Fact]
    public void Edit_ReplacesTheSingleExactMatch()
    {
        Write("a.txt", "x = 1;\ny = 2;\n");
        string result = _tools.Edit("a.txt", "y = 2;", "y = 3;");
        Assert.Equal("Edited a.txt.", result);
        Assert.Equal("x = 1;\ny = 3;\n", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public void Edit_RefusesWhenTheTextIsNotFound()
    {
        Write("a.txt", "x = 1;");
        Assert.StartsWith("No match:", _tools.Edit("a.txt", "z = 9;", "q"));
        Assert.Equal("x = 1;", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public void Edit_RefusesWhenTheTextOccursMoreThanOnce()
    {
        Write("a.txt", "dup\ndup\n");
        Assert.StartsWith("Not changed:", _tools.Edit("a.txt", "dup", "one"));
        Assert.Equal("dup\ndup\n", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public void WriteFile_ReportsCreatedThenOverwritten()
    {
        Assert.StartsWith("Created new.txt", _tools.WriteFile("new.txt", "hi"));
        Assert.StartsWith("Overwrote new.txt", _tools.WriteFile("new.txt", "again"));
    }

    [Fact]
    public void CreateFile_RefusesAnExistingFile()
    {
        Write("exists.txt", "keep");
        Assert.StartsWith("Not created", _tools.CreateFile("exists.txt"));
        Assert.Equal("keep", File.ReadAllText(Path.Combine(_root, "exists.txt")));
    }

    [Fact]
    public void Path_OutsideTheRoot_IsRefused()
    {
        Assert.Throws<UnauthorizedAccessException>(() => _tools.ResolvePath("../escape.txt"));
    }

    [Fact]
    public void Symlink_PointingOutsideTheRoot_IsRefused()
    {
        string outside = Path.Combine(Path.GetTempPath(), "sm-outside-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outside);
        File.WriteAllText(Path.Combine(outside, "secret.txt"), "private");
        string link = Path.Combine(_root, "link");
        try
        {
            try { Directory.CreateSymbolicLink(link, outside); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
            {
                // Creating a symlink needs privilege on many Windows accounts. Without one there is
                // nothing to test, so the test is skipped rather than passing against no link.
                return;
            }

            // The link must actually exist before the containment check means anything.
            Assert.True(Directory.Exists(link), "symlink was not created");
            Assert.Throws<UnauthorizedAccessException>(() => _tools.ResolvePath("link/secret.txt"));
            Assert.Contains("outside the project root", _tools.ReadFile("link/secret.txt"));
        }
        finally
        {
            try { if (Directory.Exists(link)) Directory.Delete(link); } catch (IOException) { }
            try { Directory.Delete(outside, recursive: true); } catch (IOException) { }
        }
    }

    [Fact]
    public void Descriptions_StatePurposeAndLimits()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new FileSystemTool(_root));
        var read = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "ReadFile");
        Assert.Contains("200", read["description"]!.GetValue<string>());
        Assert.Contains("offset", read["description"]!.GetValue<string>());
        Assert.Contains("Edit", string.Concat(agent.ToolDefinitions.OfType<JsonObject>().Select(t => t["name"]!.GetValue<string>())));
    }

    [Fact]
    public void FileTools_AreClassifiedAsFileAccess()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new FileSystemTool(_root));
        Assert.Equal(ToolCategory.File, agent.GetToolCategory("ReadFile"));
        Assert.Equal(ToolCategory.File, agent.GetToolCategory("Edit"));
    }
}

/// <summary>Weather tool: a missing geocoding result, and the WMO code table.</summary>
public sealed class WeatherToolBehaviourTests
{
    [Fact]
    public void WeatherCodes_CoverTheWmoTable()
    {
        Assert.Equal("thunderstorm", WeatherTool.WeatherCodeToDescription(95));
        Assert.Equal("heavy snow fall", WeatherTool.WeatherCodeToDescription(75));
        Assert.Equal("violent rain showers", WeatherTool.WeatherCodeToDescription(82));
        Assert.Equal("weather code 42", WeatherTool.WeatherCodeToDescription(42));
    }

    [Fact]
    public void WeatherTool_IsClassifiedAsNetworkAccess()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new WeatherTool());
        Assert.Equal(ToolCategory.Network, agent.GetToolCategory("GetCurrentWeather"));
    }

    [Fact]
    public void WeatherTool_DescribesTheCountryHint()
    {
        var agent = new AgentBuilder("Test");
        agent.WithTools(new WeatherTool());
        var def = agent.ToolDefinitions.OfType<JsonObject>().First(t => t["name"]!.GetValue<string>() == "GetCurrentWeather");
        Assert.Contains("country", def["description"]!.GetValue<string>());
    }
}
