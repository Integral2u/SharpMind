using SharpMind.Inference.Chat;
using SharpMind.Data.Sources;
using System.IO.Abstractions;
using System.Text.Json.Nodes;

namespace SharpMind.Tests.Security;

[Collection("Non-Parallel")]
public sealed class InterceptingFileSystemTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    private readonly InterceptingFileSystem _fs = new();
    private readonly IFileSystem _denied = NewActivated(false);
    private readonly IFileSystem _allowed = NewActivated(true);

    public void Dispose() => _dir.Dispose();

    private static readonly JsonObject NoArgs = [];

    private static InterceptingFileSystem NewActivated(bool allow)
    {
        var fs = new InterceptingFileSystem();
        fs.Activate("test_tool", NoArgs, (_, _, _, _) => Task.FromResult(allow));
        return fs;
    }

    private string PathOf(string name) => _dir.Write(name, "x");

    // ─── Every gated member must throw when the check denies ───────────────

    [Fact]
    public void AllGatedFileMembers_ThrowWhenDenied()
    {
        string p = PathOf("target.txt");
        string dest = Path.Combine(_dir.Path, "copy.txt");
        IFile f = _denied.File;

        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllBytes(p, System.Text.Encoding.UTF8.GetBytes("x")));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllBytes(p, "x"u8));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllBytes(p, System.Text.Encoding.UTF8.GetBytes("x")));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllBytes(p, "x"u8));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllLines(p, new[] { "a" }));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllLines(p, new[] { "a" }, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllText(p, "x"));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllText(p, "x", System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllText(p, "x"));
        Assert.Throws<UnauthorizedAccessException>(() => f.AppendAllText(p, "x", System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.Copy(p, dest, overwrite: true));
        Assert.Throws<UnauthorizedAccessException>(() => f.Create(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.Create(p, 4096));
        Assert.Throws<UnauthorizedAccessException>(() => f.Create(p, 4096, FileOptions.None));
        Assert.Throws<UnauthorizedAccessException>(() => f.CreateText(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.Move(p, dest, overwrite: true));
        Assert.Throws<UnauthorizedAccessException>(() => f.Open(p, FileMode.Open));
        Assert.Throws<UnauthorizedAccessException>(() => f.Open(p, FileMode.Open, FileAccess.Read));
        Assert.Throws<UnauthorizedAccessException>(() => f.Open(p, FileMode.Open, FileAccess.Read, FileShare.Read));
        Assert.Throws<UnauthorizedAccessException>(() => f.Open(p, new FileStreamOptions { Mode = FileMode.Open, Access = FileAccess.Read }));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadAllText(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadAllText(p, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadAllLines(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadAllLines(p, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadAllBytes(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadLines(p));
        Assert.Throws<UnauthorizedAccessException>(() => f.ReadLines(p, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.Replace(p, dest, null));
        Assert.Throws<UnauthorizedAccessException>(() => f.Replace(p, dest, null, ignoreMetadataErrors: true));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetAttributes(p, FileAttributes.ReadOnly));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetCreationTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetCreationTimeUtc(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetLastAccessTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetLastAccessTimeUtc(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetLastWriteTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.SetLastWriteTimeUtc(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllLines(p, new[] { "a" }));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllLines(p, new[] { "a" }, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllLines(p, (IEnumerable<string>)new[] { "a" }, System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllText(p, "x"));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllText(p, "x", System.Text.Encoding.UTF8));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllText(p, "x"));
        Assert.Throws<UnauthorizedAccessException>(() => f.WriteAllText(p, "x", System.Text.Encoding.UTF8));
    }

    [Fact]
    public async Task AllGatedAsyncFileMembers_ThrowWhenDenied()
    {
        string p = PathOf("target.txt");
        IFile f = _denied.File;

        await AssertDenied(() => f.AppendAllBytesAsync(p, new byte[] { 1 }));
        await AssertDenied(() => f.AppendAllBytesAsync(p, new byte[] { 1 }));
        await AssertDenied(() => f.AppendAllLinesAsync(p, new[] { "a" }));
        await AssertDenied(() => f.AppendAllLinesAsync(p, new[] { "a" }, System.Text.Encoding.UTF8));
        await AssertDenied(() => f.AppendAllTextAsync(p, "x"));
        await AssertDenied(() => f.AppendAllTextAsync(p, "x", System.Text.Encoding.UTF8));
        await AssertDenied(() => f.AppendAllTextAsync(p, Memory<char>.Empty));
        await AssertDenied(() => f.AppendAllTextAsync(p, Memory<char>.Empty, System.Text.Encoding.UTF8));
        await AssertDenied(() => f.ReadAllBytesAsync(p));
        await AssertDenied(() => f.ReadAllLinesAsync(p));
        await AssertDenied(() => f.ReadAllLinesAsync(p, System.Text.Encoding.UTF8));
        await AssertDenied(() => f.ReadAllTextAsync(p));
        await AssertDenied(() => f.ReadAllTextAsync(p, System.Text.Encoding.UTF8));
        await AssertDenied(() => { f.ReadLinesAsync(p); return Task.CompletedTask; });
        await AssertDenied(() => { f.ReadLinesAsync(p, System.Text.Encoding.UTF8); return Task.CompletedTask; });
        await AssertDenied(() => f.WriteAllBytesAsync(p, new byte[] { 1 }));
        await AssertDenied(() => f.WriteAllBytesAsync(p, new byte[] { 1 }));
        await AssertDenied(() => f.WriteAllLinesAsync(p, new[] { "a" }));
        await AssertDenied(() => f.WriteAllLinesAsync(p, new[] { "a" }, System.Text.Encoding.UTF8));
        await AssertDenied(() => f.WriteAllTextAsync(p, "x"));
        await AssertDenied(() => f.WriteAllTextAsync(p, "x", System.Text.Encoding.UTF8));
        await AssertDenied(() => f.WriteAllTextAsync(p, Memory<char>.Empty));
        await AssertDenied(() => f.WriteAllTextAsync(p, Memory<char>.Empty, System.Text.Encoding.UTF8));
    }

    private static async Task AssertDenied(Func<Task> call)
        => await Assert.ThrowsAsync<UnauthorizedAccessException>(async () => await call());

    [Fact]
    public void AllGatedDirectoryMembers_ThrowWhenDenied()
    {
        string p = PathOf("target.txt");
        string sub = Path.Combine(_dir.Path, "sub");
        IDirectory d = _denied.Directory;

        Assert.Throws<UnauthorizedAccessException>(() => d.CreateDirectory(sub));
        Assert.Throws<UnauthorizedAccessException>(() => d.Delete(p));
        Assert.Throws<UnauthorizedAccessException>(() => d.Delete(sub, recursive: true));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateDirectories(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFiles(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFiles(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFiles(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFiles(_dir.Path, "*.txt", new EnumerationOptions()));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateDirectories(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateDirectories(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFileSystemEntries(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFileSystemEntries(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.EnumerateFileSystemEntries(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetDirectories(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetDirectories(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetDirectories(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFiles(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFiles(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFiles(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFileSystemEntries(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFileSystemEntries(_dir.Path, "*.txt"));
        Assert.Throws<UnauthorizedAccessException>(() => d.GetFileSystemEntries(_dir.Path, "*.txt", SearchOption.TopDirectoryOnly));
        Assert.Throws<UnauthorizedAccessException>(() => d.Move(sub, sub + "2"));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetCurrentDirectory(_dir.Path));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetCreationTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetCreationTimeUtc(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetLastAccessTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetLastAccessTimeUtc(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetLastWriteTime(p, DateTime.UtcNow));
        Assert.Throws<UnauthorizedAccessException>(() => d.SetLastWriteTimeUtc(p, DateTime.UtcNow));
    }

    // ─── Permitted members actually perform IO; metadata stays ungated ─────

    [Fact]
    public void PermittedMember_ReadsAndWrites()
    {
        string p = PathOf("target.txt");
        IFile f = _allowed.File;

        f.WriteAllText(p, "hello");
        Assert.Equal("hello", f.ReadAllText(p));

        f.AppendAllText(p, " world");
        Assert.Equal("hello world", f.ReadAllText(p));

        f.SetAttributes(p, FileAttributes.Normal);
        Assert.True(f.Exists(p));
        Assert.Equal(System.Text.Encoding.UTF8.GetBytes("hello world"), f.ReadAllBytes(p));
    }

    [Fact]
    public void DeniedGate_OnlyAppliesWhileActivated()
    {
        string p = PathOf("target.txt");
        var fs = NewActivated(true);

        fs.File.WriteAllText(p, "ok");
        Assert.Equal("ok", fs.File.ReadAllText(p));

        fs.Deactivate();
        // Gating is off entirely — ordinary session IO passes through.
        fs.File.WriteAllText(p, "after");
        Assert.Equal("after", fs.File.ReadAllText(p));
    }

    [Fact]
    public void UngatedMetadataMembers_DoNotThrow_WhenDenied()
    {
        string p = PathOf("target.txt");
        IFile f = _denied.File;
        IDirectory d = _denied.Directory;

        try
        {
            Assert.False(f.Exists(System.IO.Path.Combine(_dir.Path, "nope.txt")));
            _ = f.GetAttributes(p);
            _ = f.GetCreationTime(p);
            _ = f.GetLastAccessTime(p);
            _ = f.GetLastWriteTime(p);
            _ = f.GetCreationTimeUtc(p);
            _ = d.GetCurrentDirectory();
            _ = d.GetLogicalDrives();
            _ = d.GetParent(_dir.Path);
            _ = d.GetDirectoryRoot(_dir.Path);
            _ = d.Exists(_dir.Path);
        }
        catch (UnauthorizedAccessException)
        {
            Assert.Fail("metadata getters must not be gated");
        }
    }
}