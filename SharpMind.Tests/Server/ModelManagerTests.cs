using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Server;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Server;

public class ModelManagerTests
{
    [Fact]
    public void ResolvedModelsDir_FallsBackToUserProfile()
    {
        var options = new SharpMindServerOptions { ModelsDir = "" };
        var expected = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "SharpMind", "Models");
        Assert.Equal(expected, options.ResolvedModelsDir);
    }

    [Fact]
    public void ResolvedModelsDir_UsesExplicitPath()
    {
        var options = new SharpMindServerOptions { ModelsDir = @"D:\CustomModels" };
        Assert.Equal(@"D:\CustomModels", options.ResolvedModelsDir);
    }

    [Fact]
    public void ModelManager_DirectoryScan_FindsModels()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            // Create dummy model files
            File.WriteAllBytes(Path.Combine(dir, "model1.gguf"), [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(dir, "model2.smm"), [4, 5, 6]);
            File.WriteAllText(Path.Combine(dir, "readme.txt"), "not a model");

            var options = new SharpMindServerOptions { ModelsDir = dir };
            using var manager = new ModelManager(options);

            var models = manager.GetAvailableModels();

            // Should find .gguf and .smm files, not .txt
            var modelIds = models.Select(m => m.ModelId).ToList();
            Assert.Contains("model1.gguf", modelIds);
            Assert.Contains("model2.smm", modelIds);
            Assert.DoesNotContain("readme.txt", modelIds);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ModelManager_GetModelInfo_ReturnsNullForMissing()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            var options = new SharpMindServerOptions { ModelsDir = dir };
            using var manager = new ModelManager(options);

            Assert.Null(manager.GetModelInfo("nonexistent.gguf"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    /// <summary>
    /// The folder watcher rescans on every Deleted event, on a thread-pool thread. Deleting the
    /// models folder raises one event per file and then removes the folder under those scans;
    /// the scan's DirectoryNotFoundException was unhandled there and took the process down.
    /// When this regresses the test host crashes rather than the test failing.
    /// </summary>
    [Fact]
    public void ModelManager_SurvivesItsModelsFolderBeingDeleted()
    {
        for (int round = 0; round < 10; round++)
        {
            var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            for (int i = 0; i < 50; i++)
                File.WriteAllBytes(Path.Combine(dir, $"model{i}.gguf"), [1, 2, 3]);

            using var manager = new ModelManager(new SharpMindServerOptions { ModelsDir = dir });
            Directory.Delete(dir, true);
            Thread.Sleep(100); // let the queued watcher events run while the manager is alive

            Assert.False(Directory.Exists(dir));
        }
    }

    [Fact]
    public void ModelManager_Unload_ReturnsFalseForMissing()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            var options = new SharpMindServerOptions { ModelsDir = dir };
            using var manager = new ModelManager(options);

            Assert.False(manager.Unload("nonexistent.gguf"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    [Fact]
    public void ModelManager_ModelInfo_HasCreatedUnixTimestamp()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            File.WriteAllBytes(Path.Combine(dir, "test.gguf"), [1, 2, 3]);

            var options = new SharpMindServerOptions { ModelsDir = dir };
            using var manager = new ModelManager(options);

            var info = manager.GetModelInfo("test.gguf");
            Assert.NotNull(info);
            Assert.True(info!.CreatedUnix > 0);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    /// <summary>
    /// P1: Unload raced in-flight requests — it disposed the Transformer while a
    /// request still held its LoadAsync reference, then the request ran on a
    /// disposed model. Idle models settle at RefCount 0 inside the manager (the
    /// loader marks them KeepAlive), so Unload is only legal there; an
    /// outstanding reference must refuse.
    /// </summary>
    [Fact]
    public async Task ModelManager_Unload_RefusesWhileReferenceHeld_ThenSucceeds()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            ExportTinyModel(Path.Combine(dir, "tiny.smm"));

            using var manager = new ModelManager(new SharpMindServerOptions { ModelsDir = dir });

            var loaded = await manager.LoadAsync("tiny.smm");
            Assert.NotNull(loaded);

            // Held reference -> must NOT unload (this used to dispose mid-request).
            Assert.False(manager.Unload("tiny.smm"));
            Assert.NotNull(manager.GetLoaded("tiny.smm"));

            // Release the hold; model is KeepAlive so it stays resident but idle.
            manager.Release("tiny.smm");
            Assert.NotNull(manager.GetLoaded("tiny.smm"));

            Assert.True(manager.Unload("tiny.smm"));
            Assert.Null(manager.GetLoaded("tiny.smm"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    /// <summary>
    /// Preload warms the model then releases its hold, so a preloaded model is
    /// still loadable — and its warm-up reference must not have been leaked.
    /// </summary>
    [Fact]
    public async Task ModelManager_Preload_LeavesModelLoadableAndUnloadable()
    {
        var dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);

        try
        {
            ExportTinyModel(Path.Combine(dir, "tiny.smm"));

            using var manager = new ModelManager(new SharpMindServerOptions { ModelsDir = dir });

            Assert.True(await manager.PreloadAsync("tiny.smm"));
            var loaded = await manager.LoadAsync("tiny.smm");
            Assert.NotNull(loaded);
            manager.Release("tiny.smm");
            Assert.True(manager.Unload("tiny.smm"));
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }

    private static void ExportTinyModel(string smmPath)
    {
        var cfg = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        using var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);

        var tokens = new string[cfg.VocabSize];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = Vocabulary.ByteTokenString(i);
        var tokenizer = Tokenizer.FromGguf(tokens, merges: null, tokenTypes: null, bosId: -1, eosId: -1);

        SmmTrainingExporter.Export(weights, tokenizer, smmPath, new SmmWriteOptions { Source = "training" });
    }
}
