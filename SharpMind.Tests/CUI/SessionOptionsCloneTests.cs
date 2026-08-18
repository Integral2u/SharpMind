using SharpMind.Core;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Model.Config;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Guards the shared copy logic behind <see cref="SessionOptions.Clone"/>. The
/// CUI's launch path clones options before building a session (MainWindow clones
/// at every launch/resume), so a field that fails to copy here is silently
/// dropped at launch — exactly what happened to the MaxTokens/SkipAgentPrompt/
/// DisableTools knobs before they were routed through this method.
/// </summary>
public sealed class SessionOptionsCloneTests
{
    [Fact]
    public void Clone_PreservesEveryField_IncludingNewKnobsAndUserName()
    {
        var source = new SessionOptions
        {
            ModelPath = "m.gguf",
            ProjectPath = "C:\\proj",
            SkillFolders = ["s1", "s2"],
            ToolAssemblyPaths = ["t1.dll"],
            ToolsFolder = "tools",
            Generator = GeneratorStrategy.Medusa,
            Cache = CacheStrategy.Paged,
            Formatter = FormatterStrategy.ChatML,
            LoadMode = LoadMode.Streaming,
            HardwareTier = HardwareTier.AVX2,
            UseGpu = true,
            GpuNonQuant = false,
            GpuVecDot = true,
            GpuMatMul = true,
            UseParallelKernels = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Never,
            Sampling = new SamplingConfig { Temperature = 0.4f, TopK = 5 },
            Generation = new GenerationConfig { MaxNewTokens = 128 },
            MaxTokens = 255,
            AgentName = "Bob",
            UserName = "Alice",
            Compactor = CompactorStrategy.Summarize,
            PluginCompactorName = "plug",
            AgentsEnabled = true,
            MaxAgentDepth = 3,
            MaxToolCallsPerTurn = 4,
            ShowThinking = false,
            EnableThinking = true,
            DisabledTools = ["a", "b"],
            DisabledPreProcessors = ["p"],
            DisabledPostProcessors = ["q"],
            SkipAgentPrompt = true,
            DisableTools = true,
        };

        var clone = source.Clone();

        Assert.Equal(source.ModelPath, clone.ModelPath);
        Assert.Equal(source.ProjectPath, clone.ProjectPath);
        Assert.Equal(source.SkillFolders, clone.SkillFolders);
        Assert.Equal(source.ToolAssemblyPaths, clone.ToolAssemblyPaths);
        Assert.Equal(source.ToolsFolder, clone.ToolsFolder);
        Assert.Equal(source.Generator, clone.Generator);
        Assert.Equal(source.Cache, clone.Cache);
        Assert.Equal(source.Formatter, clone.Formatter);
        Assert.Equal(source.LoadMode, clone.LoadMode);
        Assert.Equal(source.HardwareTier, clone.HardwareTier);
        Assert.Equal(source.UseGpu, clone.UseGpu);
        Assert.Equal(source.GpuNonQuant, clone.GpuNonQuant);
        Assert.Equal(source.GpuVecDot, clone.GpuVecDot);
        Assert.Equal(source.GpuMatMul, clone.GpuMatMul);
        Assert.Equal(source.UseParallelKernels, clone.UseParallelKernels);
        Assert.Equal(source.FileAccess, clone.FileAccess);
        Assert.Equal(source.NetworkAccess, clone.NetworkAccess);
        Assert.Equal(source.Sampling, clone.Sampling);
        Assert.Equal(source.Generation, clone.Generation);
        Assert.Equal(255, clone.MaxTokens);
        Assert.Equal("Bob", clone.AgentName);
        Assert.Equal("Alice", clone.UserName);
        Assert.Equal(source.Compactor, clone.Compactor);
        Assert.Equal("plug", clone.PluginCompactorName);
        Assert.True(clone.AgentsEnabled);
        Assert.Equal(3, clone.MaxAgentDepth);
        Assert.Equal(4, clone.MaxToolCallsPerTurn);
        Assert.False(clone.ShowThinking);
        Assert.True(clone.EnableThinking);
        Assert.Equal(["a", "b"], clone.DisabledTools);
        Assert.Equal(["p"], clone.DisabledPreProcessors);
        Assert.Equal(["q"], clone.DisabledPostProcessors);
        Assert.True(clone.SkipAgentPrompt);
        Assert.True(clone.DisableTools);
    }

    [Fact]
    public void Clone_CollectionCopiesAreDeep()
    {
        var source = new SessionOptions { DisabledTools = ["x"], SkillFolders = ["s"] };
        var clone = source.Clone();

        clone.DisabledTools.Add("y");
        clone.SkillFolders.Add("extra");

        Assert.Single(source.DisabledTools);
        Assert.Single(source.SkillFolders);
    }

    [Fact]
    public void Clone_DoesNotShareCollectionInstances()
    {
        var source = new SessionOptions { DisabledTools = ["x"], SkillFolders = ["s"] };
        var clone = source.Clone();

        Assert.NotSame(source.DisabledTools, clone.DisabledTools);
        Assert.NotSame(source.SkillFolders, clone.SkillFolders);
    }
}
