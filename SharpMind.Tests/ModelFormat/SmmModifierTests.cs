using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Verifies <see cref="SmmModifier"/>/<see cref="SmmModelDocument"/>: reading
/// and rewriting the system prompt, skills and plugins of an .SMM container
/// while preserving the tensor data region and index byte-for-byte.
/// </summary>
public sealed class SmmModifierTests : IDisposable
{
    private readonly TempDirectory _temp = new();
    public void Dispose() => _temp.Dispose();

    private static ModelConfig Config => new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 8,
        NormEps = 1e-3f,
    };

    private static SmmTensorData Tensor(string name, int count, int seed)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = (float)(rng.NextDouble() * 2 - 1);
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return new SmmTensorData { Name = name, Shape = [count], Dtype = QuantDType.F32, GetBytes = () => bytes };
    }

    private static SmmWriteOptions Options(string? prompt = null, string[]? skills = null, SmmPluginEntry[]? plugins = null)
        => new()
        {
            Source = "test",
            SystemPrompt = prompt,
            Skills = skills,
            Plugins = plugins,
            Outputs = plugins is { Length: > 0 } ? SmmOutputs.Default | SmmOutputs.Plugins : SmmOutputs.Default,
        };

    private string WriteSmm(string name, SmmWriteOptions? options = null)
    {
        string path = Path.Combine(_temp.Path, name);
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Tensor("token_embd.weight", 64, seed: 1),
            Tensor("output_norm.weight", 16, seed: 2),
        ], options);
        return path;
    }

    [Fact]
    public void Read_ReturnsEmptyFields_WhenNothingEmbedded()
    {
        string path = WriteSmm("plain.smm");

        var doc = SmmModifier.Read(path);

        Assert.Equal("", doc.SystemPrompt);
        Assert.Empty(doc.Skills);
        Assert.Empty(doc.Plugins);
    }

    [Fact]
    public void Read_ReturnsEmbeddedFields()
    {
        byte[] asm = [1, 2, 3, 4, 5];
        string path = WriteSmm("embedded.smm", Options(
            prompt: "Assistant prompt.",
            skills: ["# Skill A", "# Skill B"],
            plugins: [new SmmPluginEntry { Name = "Tool.dll", AssemblyBytes = asm, Recommended = true }]));

        var doc = SmmModifier.Read(path);

        Assert.Equal("Assistant prompt.", doc.SystemPrompt);
        Assert.Equal(["# Skill A", "# Skill B"], doc.Skills);
        var plugin = Assert.Single(doc.Plugins);
        Assert.Equal("Tool.dll", plugin.Name);
        Assert.True(plugin.Recommended);
        Assert.Equal(asm, plugin.AssemblyBytes);
    }

    [Fact]
    public void Write_UpdatesAllFields_AndPreservesTensorBytes()
    {
        string path = WriteSmm("update.smm", Options(prompt: "Old prompt.", skills: ["# Old skill"]));
        byte[] before = SnapshotTensors(path);

        var doc = SmmModifier.Read(path);
        doc.SystemPrompt = "New system prompt.";
        doc.Skills.Clear();
        doc.Skills.Add("# New skill 1");
        doc.Skills.Add("# New skill 2");
        byte[] pluginAsm = [9, 8, 7];
        doc.Plugins.Add(new SmmPluginEntry { Name = "Plug.dll", AssemblyBytes = pluginAsm });

        SmmModifier.Write(path, doc);

        Assert.Equal("New system prompt.", SmmLoader.LoadSystemPrompt(path));
        Assert.Equal(["# New skill 1", "# New skill 2"], SmmLoader.LoadSkills(path));
        var plugin = Assert.Single(SmmLoader.LoadPlugins(path));
        Assert.Equal("Plug.dll", plugin.Name);
        Assert.Equal(pluginAsm, plugin.AssemblyBytes);

        AssertTensorBytesPreserved(path, before);
    }

    [Fact]
    public void Write_ClearsAllFields_AndPreservesTensorBytes()
    {
        byte[] asm = [1, 2, 3];
        string path = WriteSmm("clear.smm", Options(
            prompt: "Remove me.",
            skills: ["# Remove me too."],
            plugins: [new SmmPluginEntry { Name = "Old.dll", AssemblyBytes = asm }]));
        byte[] before = SnapshotTensors(path);

        var doc = SmmModifier.Read(path);
        doc.SystemPrompt = "";
        doc.Skills.Clear();
        doc.Plugins.Clear();
        SmmModifier.Write(path, doc);

        Assert.Null(SmmLoader.LoadSystemPrompt(path));
        Assert.Empty(SmmLoader.LoadSkills(path));
        Assert.Empty(SmmLoader.LoadPlugins(path));
        AssertTensorBytesPreserved(path, before);
    }

    [Fact]
    public void Write_PreservesChatTemplateAndConfig()
    {
        string path = WriteSmm("template.smm");
        // Re-write through the writer with a chat template so the meta carries it.
        File.Delete(path);
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: "{{- messages }}",
            tensors: [Tensor("token_embd.weight", 64, seed: 1)], options: Options());
        byte[] before = SnapshotTensors(path);

        var doc = SmmModifier.Read(path);
        doc.SystemPrompt = "Set after template.";
        SmmModifier.Write(path, doc);

        var meta = SmmLoader.LoadMeta(path);
        Assert.Equal("{{- messages }}", meta.GetChatTemplate());
        Assert.NotNull(SmmLoader.LoadConfig(meta));
        AssertTensorBytesPreserved(path, before);
    }

    [Fact]
    public void Write_MissingSourceFile_FailsWithoutLeavingTemp()
    {
        string path = Path.Combine(_temp.Path, "absent.smm");

        var ex = Assert.ThrowsAny<Exception>(() => SmmModifier.Write(path, new SmmModelDocument()));
        Assert.NotNull(ex);
        Assert.False(File.Exists(path + ".tmp"));
    }

    [Fact]
    public void Read_MissingSourceFile_Throws()
    {
        string path = Path.Combine(_temp.Path, "absent.smm");
        Assert.Throws<FileNotFoundException>(() => SmmModifier.Read(path));
    }

    [Fact]
    public void Write_RepeatedNoOp_WritesSucceed()
    {
        string path = WriteSmm("repeat.smm");
        byte[] before = SnapshotTensors(path);

        var doc = SmmModifier.Read(path);
        SmmModifier.Write(path, doc);
        SmmModifier.Write(path, doc);
        SmmModifier.Write(path, doc);

        Assert.NotNull(SmmLoader.LoadMeta(path));
        AssertTensorBytesPreserved(path, before);
    }

    [Fact]
    public void Write_SecondRealEdit_AfterFirstEdit_Succeeds()
    {
        string path = WriteSmm("twoedits.smm");

        var first = SmmModifier.Read(path);
        first.SystemPrompt = "First edit.";
        SmmModifier.Write(path, first);

        var second = SmmModifier.Read(path);
        second.SystemPrompt = "Second edit.";
        SmmModifier.Write(path, second);

        Assert.Equal("Second edit.", SmmLoader.LoadSystemPrompt(path));
    }

    [Fact]
    public void Write_NoOpEdit_LeavesFileByteForByteUntouched()
    {
        string path = WriteSmm("noop.smm");
        DateTime beforeTimestamp = File.GetLastWriteTimeUtc(path);

        var doc = SmmModifier.Read(path);
        SmmModifier.Write(path, doc);

        Assert.Equal(beforeTimestamp, File.GetLastWriteTimeUtc(path));
        Assert.False(File.Exists(path + ".tmp"));

        // Whole-file compare: re-reading the untouched file and saving it again
        // must also be a no-op (identical bytes, timestamp untouched, no tmp).
        byte[] before = File.ReadAllBytes(path);
        var sameDoc = SmmModifier.Read(path);
        SmmModifier.Write(path, sameDoc);
        Assert.Equal(before, File.ReadAllBytes(path));
        Assert.Equal(beforeTimestamp, File.GetLastWriteTimeUtc(path));
    }

    private static byte[] SnapshotTensors(string path)
    {
        var entries = SmmLoader.ReadTensorIndex(path);
        using var ms = new MemoryStream();
        foreach (var entry in entries)
        {
            long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
            byte[] bytes = SmmLoader.ReadTensorBytes(path, entry, rawSize);
            ms.Write(BitConverter.GetBytes(entry.Name.Length), 0, 4);
            var name = System.Text.Encoding.UTF8.GetBytes(entry.Name);
            ms.Write(name, 0, name.Length);
            ms.Write(BitConverter.GetBytes(rawSize), 0, 8);
            ms.Write(bytes, 0, bytes.Length);
        }
        return ms.ToArray();
    }

    private static void AssertTensorBytesPreserved(string path, byte[] before)
        => Assert.Equal(before, SnapshotTensors(path));
}