using SharpMind.Server;

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
}
