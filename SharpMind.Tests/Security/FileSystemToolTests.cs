using SharpMind.Core.AgentTools;
using Xunit;

namespace SharpMind.Tests.Security;

/// <summary>
/// The FileSystemTool sandbox must reject any path that resolves outside the
/// project root. ResolvePath compares against the prefix — without a directory
/// separator boundary a sibling whose name merely starts with the root's name
/// (e.g. "Proj" root vs "ProjEvil" sibling) slips through.
/// </summary>
public sealed class FileSystemToolTests
{
    private sealed record Fixture(string Base, string Root, string Evil) : IDisposable
    {
        public static Fixture Create()
        {
            string baseDir = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                System.IO.Path.GetRandomFileName());
            Directory.CreateDirectory(baseDir);
            string root = System.IO.Path.Combine(baseDir, "Proj");
            string evil = System.IO.Path.Combine(baseDir, "ProjEvil");
            Directory.CreateDirectory(root);
            Directory.CreateDirectory(evil);
            return new Fixture(baseDir, root, evil);
        }

        public void Dispose()
        {
            if (Directory.Exists(Base)) Directory.Delete(Base, recursive: true);
        }
    }

    [Fact]
    public void SiblingSharingRootNamePrefix_IsRejected()
    {
        using var fx = Fixture.Create();
        string secret = "top secret";
        File.WriteAllText(System.IO.Path.Combine(fx.Evil, "secret.txt"), secret);

        var tool = new FileSystemTool(fx.Root);

        // "../ProjEvil/secret.txt" normalises inside a sibling whose name starts
        // with the root's name — the old prefix-only StartsWith check let this
        // read escape the sandbox.
        string result = tool.ReadFile(".." + System.IO.Path.DirectorySeparatorChar
            + "ProjEvil" + System.IO.Path.DirectorySeparatorChar + "secret.txt");

        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(secret, result, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void SiblingSharingRootNamePrefix_WriteIsRejected()
    {
        using var fx = Fixture.Create();

        var tool = new FileSystemTool(fx.Root);

        string result = tool.WriteFile(".." + System.IO.Path.DirectorySeparatorChar
            + "ProjEvil" + System.IO.Path.DirectorySeparatorChar + "planted.txt", "owned");

        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
        Assert.False(System.IO.File.Exists(System.IO.Path.Combine(fx.Evil, "planted.txt")));
    }

    [Fact]
    public void InsideRoot_PathsRemainAllowed()
    {
        using var fx = Fixture.Create();
        File.WriteAllText(System.IO.Path.Combine(fx.Root, "notes.txt"), "hello from the sandbox");

        var tool = new FileSystemTool(fx.Root);

        string result = tool.ReadFile("notes.txt");
        Assert.Equal("hello from the sandbox", result);
    }

    [Fact]
    public void DirectlyToParent_IsRejected()
    {
        using var fx = Fixture.Create();
        File.WriteAllText(System.IO.Path.Combine(fx.Base, "outside.txt"), "outside world");

        var tool = new FileSystemTool(fx.Root);

        string result = tool.ReadFile(".." + System.IO.Path.DirectorySeparatorChar + "outside.txt");
        Assert.Contains("Error", result, StringComparison.OrdinalIgnoreCase);
    }
}