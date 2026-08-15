using SharpMind.CUI.App;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Verifies the default user-folder layout (<see cref="SharpMindPaths"/>) and
/// the AppSettings export/model-folder resolution rules that drive where new
/// training jobs default their export path.
/// </summary>
public sealed class SharpMindPathsTests
{
    [Fact]
    public void DefaultFolders_LiveUnderOverrideRoot()
    {
        string root = Path.Combine(Path.GetTempPath(), "sm-paths-" + Guid.NewGuid().ToString("N"));
        SharpMindPaths.OverrideRoot = root;

        try
        {
            Assert.Equal(Path.Combine(root, "SharpMind"), SharpMindPaths.Root);
            Assert.Equal(Path.Combine(root, "SharpMind", "Training"), SharpMindPaths.Training);
            Assert.Equal(Path.Combine(root, "SharpMind", "Chat Sessions"), SharpMindPaths.ChatSessions);
            Assert.Equal(Path.Combine(root, "SharpMind", "Models"), SharpMindPaths.Models);
            Assert.Equal(Path.Combine(root, "SharpMind", "Corpus"), SharpMindPaths.Corpus);
            Assert.Equal(SharpMindPaths.Training, TrainJobSettings.DefaultFolder);
            Assert.Equal(SharpMindPaths.ChatSessions, SavedSession.DefaultFolder);

            SharpMindPaths.EnsureCreated();
            Assert.True(Directory.Exists(SharpMindPaths.Training));
            Assert.True(Directory.Exists(SharpMindPaths.ChatSessions));
            Assert.True(Directory.Exists(SharpMindPaths.Models));
            Assert.True(Directory.Exists(SharpMindPaths.Corpus));
        }
        finally
        {
            SharpMindPaths.OverrideRoot = null;
            try { Directory.Delete(root, recursive: true); } catch { }
        }
    }

    [Fact]
    public void ResolvedExportFolder_FallsBackToModelFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "sm-settings-" + Guid.NewGuid().ToString("N"));
        SharpMindPaths.OverrideRoot = root;
        try
        {
            var settings = new AppSettings();
            // Nothing recorded yet → the default Models folder.
            Assert.Equal(SharpMindPaths.Models, settings.ResolvedExportFolder);
            Assert.Equal(SharpMindPaths.Models, settings.ResolvedModelFolder);

            // A recorded model folder is the fallback when no export path exists.
            settings.DefaultModelFolder = @"C:\my-models";
            Assert.Equal(@"C:\my-models", settings.ResolvedExportFolder);

            // A recorded export path wins outright.
            settings.LastExportPath = @"C:\exported";
            Assert.Equal(@"C:\exported", settings.ResolvedExportFolder);
        }
        finally
        {
            SharpMindPaths.OverrideRoot = null;
            try { Directory.Delete(root, recursive: true); } catch { }
        }
    }

    [Fact]
    public void NewJob_DefaultsExportPathToResolvedExportFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "sm-job-" + Guid.NewGuid().ToString("N"));
        SharpMindPaths.OverrideRoot = root;
        try
        {
            // No export path recorded → the default Models folder pre-fills the job.
            var settings = new AppSettings { DefaultModelFolder = @"C:\models" };
            var job = new TrainJobSettings { ExportPath = settings.ResolvedExportFolder };
            Assert.Equal(@"C:\models", job.ExportPath);
            Assert.Equal(@"C:\models", job.ExportFolder);

            // A recorded export path wins.
            settings.LastExportPath = @"C:\exported";
            var job2 = new TrainJobSettings { ExportPath = settings.ResolvedExportFolder };
            Assert.Equal(@"C:\exported", job2.ExportPath);
        }
        finally
        {
            SharpMindPaths.OverrideRoot = null;
            try { Directory.Delete(root, recursive: true); } catch { }
        }
    }

    [Fact]
    public void DefaultRoot_LivesUnderUserProfile_NotDocuments()
    {
        // No override → the default tree sits at the profile root, beside Documents.
        // This guards the Windows gotchas (Controlled Folder Access, OneDrive Known
        // Folder Move) that Documents-based defaults would hit.
        SharpMindPaths.OverrideRoot = null;
        string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        Assert.Equal(Path.Combine(profile, "SharpMind"), SharpMindPaths.Root);
        Assert.Equal(Path.Combine(profile, "SharpMind", "Models"), SharpMindPaths.Models);
        Assert.Equal(Path.Combine(profile, "SharpMind", "Corpus"), SharpMindPaths.Corpus);
        Assert.False(SharpMindPaths.Root.StartsWith(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), StringComparison.OrdinalIgnoreCase));
    }
}