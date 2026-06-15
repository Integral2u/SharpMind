using Microsoft.Win32.SafeHandles;
using SharpMind.Inference.Agent;
using System.IO.Abstractions;
using System.Runtime.Versioning;
using System.Text;
using System.Text.Json.Nodes;

namespace SharpMind.Inference.Chat;

// File system interceptor

/// <summary>
/// <see cref="IFileSystem"/> decorator that gates every file-system operation
/// through <see cref="IoPermissionCheck"/> while a tool call is in flight.
/// <para>
/// Wrap the real <see cref="IFileSystem"/> (e.g. <c>new FileSystem()</c>) and pass
/// this instance wherever file IO is needed. ChatSession enables and disables it
/// around each <see cref="IAgentBuilder.CallToolAsync"/> call so that ordinary
/// session IO is never blocked.
/// </para>
/// </summary>
public sealed class InterceptingFileSystem : IFileSystem
{
    private readonly FileSystem _inner = new();
    private IoPermissionCheck? _check;
    private string _currentTool = string.Empty;
    private JsonObject _currentArgs = [];

    // Internal activation (ChatSession only)

    internal void Activate(string toolName, JsonObject args, IoPermissionCheck check)
    {
        _currentTool = toolName;
        _currentArgs = args;
        _check = check;
    }

    internal void Deactivate()
    {
        _check = null;
        _currentTool = string.Empty;
        _currentArgs = [];
    }

    // Gate helper

    /// <summary>
    /// Synchronously blocks until the async permission check resolves.
    /// File-system APIs are synchronous, so we must bridge here.
    /// Throws <see cref="UnauthorizedAccessException"/> when access is denied.
    /// </summary>
    private void Gate(string path)
    {
        if (_check is null) return;
        bool ok = _check(_currentTool, ToolCategory.File, path, _currentArgs)
                      .GetAwaiter().GetResult();
        if (!ok)
            throw new UnauthorizedAccessException(
                $"Tool '{_currentTool}' was denied file access to '{path}'.");
    }

    // IFileSystem passthrough with gate
    // Only the most commonly used surface members need gating; everything else
    // delegates to the inner implementation transparently.

    public IFile File => new GatedFile(_inner.File, Gate);
    public IDirectory Directory => new GatedDirectory(_inner.Directory, Gate);
    public IPath Path => _inner.Path;
    public IFileInfoFactory FileInfo => _inner.FileInfo;
    public IFileStreamFactory FileStream => _inner.FileStream;
    public IDirectoryInfoFactory DirectoryInfo => _inner.DirectoryInfo;
    public IDriveInfoFactory DriveInfo => _inner.DriveInfo;
    public IFileSystemWatcherFactory FileSystemWatcher => _inner.FileSystemWatcher;
    public IFileVersionInfoFactory FileVersionInfo => _inner.FileVersionInfo;

    // Gated IFile

    private sealed class GatedFile(IFile inner, Action<string> gate) : IFile
    {
        public IFileSystem FileSystem => inner.FileSystem;

        public string ReadAllText(string path) { gate(path); return inner.ReadAllText(path); }
        public string ReadAllText(string path, System.Text.Encoding encoding) { gate(path); return inner.ReadAllText(path, encoding); }
        public void WriteAllText(string path, string? contents) { gate(path); inner.WriteAllText(path, contents); }
        public void WriteAllText(string path, string? contents, System.Text.Encoding encoding) { gate(path); inner.WriteAllText(path, contents, encoding); }
        public byte[] ReadAllBytes(string path) { gate(path); return inner.ReadAllBytes(path); }
        public void WriteAllBytes(string path, byte[] bytes) { gate(path); inner.WriteAllBytes(path, bytes); }
        public void AppendAllText(string path, string? contents) { gate(path); inner.AppendAllText(path, contents); }
        public void Delete(string path) { gate(path); inner.Delete(path); }
        public bool Exists(string? path) => inner.Exists(path);
        public void Copy(string sourceFileName, string destFileName) { gate(sourceFileName); gate(destFileName); inner.Copy(sourceFileName, destFileName); }
        public void Move(string sourceFileName, string destFileName) { gate(sourceFileName); gate(destFileName); inner.Move(sourceFileName, destFileName); }
        public FileSystemStream OpenRead(string path) { gate(path); return inner.OpenRead(path); }
        public FileSystemStream OpenWrite(string path) { gate(path); return inner.OpenWrite(path); }
        public FileSystemStream Open(string path, FileMode mode) { gate(path); return inner.Open(path, mode); }
        public FileSystemStream Create(string path) { gate(path); return inner.Create(path); }
        public StreamReader OpenText(string path) { gate(path); return inner.OpenText(path); }
        public StreamWriter CreateText(string path) { gate(path); return inner.CreateText(path); }
        public StreamWriter AppendText(string path) { gate(path); return inner.AppendText(path); }
        public string[] ReadAllLines(string path) { gate(path); return inner.ReadAllLines(path); }
        public void WriteAllLines(string path, IEnumerable<string> contents) { gate(path); inner.WriteAllLines(path, contents); }
        public IEnumerable<string> ReadLines(string path) { gate(path); return inner.ReadLines(path); }
        public FileAttributes GetAttributes(string path) => inner.GetAttributes(path);
        public void SetAttributes(string path, FileAttributes fileAttributes) => inner.SetAttributes(path, fileAttributes);
        public DateTime GetCreationTime(string path) => inner.GetCreationTime(path);
        public DateTime GetLastWriteTime(string path) => inner.GetLastWriteTime(path);
        public DateTime GetLastAccessTime(string path) => inner.GetLastAccessTime(path);

        // Async variants — gate then delegate
        public Task<string> ReadAllTextAsync(string path, CancellationToken ct = default)
        { gate(path); return inner.ReadAllTextAsync(path, ct); }
        public Task WriteAllTextAsync(string path, string? contents, CancellationToken ct = default)
        { gate(path); return inner.WriteAllTextAsync(path, contents, ct); }
        public Task<byte[]> ReadAllBytesAsync(string path, CancellationToken ct = default)
        { gate(path); return inner.ReadAllBytesAsync(path, ct); }
        public Task WriteAllBytesAsync(string path, byte[] bytes, CancellationToken ct = default)
        { gate(path); return inner.WriteAllBytesAsync(path, bytes, ct); }

        public void AppendAllBytes(string path, byte[] bytes) => inner.AppendAllBytes(path, bytes);

        public void AppendAllBytes(string path, ReadOnlySpan<byte> bytes) => inner.AppendAllBytes(path, bytes);

        public Task AppendAllBytesAsync(string path, byte[] bytes, CancellationToken cancellationToken = default) => inner.AppendAllBytesAsync(path, bytes, cancellationToken);

        public Task AppendAllBytesAsync(string path, ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken = default) => inner.AppendAllBytesAsync(path, bytes, cancellationToken);

        public void AppendAllLines(string path, IEnumerable<string> contents) => inner.AppendAllLines(path, contents);

        public void AppendAllLines(string path, IEnumerable<string> contents, Encoding encoding) => inner.AppendAllLines(path, contents, encoding);

        public Task AppendAllLinesAsync(string path, IEnumerable<string> contents, CancellationToken cancellationToken = default) => inner.AppendAllLinesAsync(path, contents, cancellationToken);

        public Task AppendAllLinesAsync(string path, IEnumerable<string> contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.AppendAllLinesAsync(path, contents, encoding, cancellationToken);

        public void AppendAllText(string path, string? contents, Encoding encoding) => inner.AppendAllText(path, contents, encoding);

        public void AppendAllText(string path, ReadOnlySpan<char> contents) => inner.AppendAllText(path, contents);

        public void AppendAllText(string path, ReadOnlySpan<char> contents, Encoding encoding) => inner.AppendAllText(path, contents, encoding);

        public Task AppendAllTextAsync(string path, string? contents, CancellationToken cancellationToken = default) => inner.AppendAllTextAsync(path, contents, cancellationToken);

        public Task AppendAllTextAsync(string path, string? contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.AppendAllTextAsync(path, contents, encoding, cancellationToken);

        public Task AppendAllTextAsync(string path, ReadOnlyMemory<char> contents, CancellationToken cancellationToken = default) => inner.AppendAllTextAsync(path, contents, cancellationToken);

        public Task AppendAllTextAsync(string path, ReadOnlyMemory<char> contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.AppendAllTextAsync(path, contents, encoding, cancellationToken);

        public void Copy(string sourceFileName, string destFileName, bool overwrite) => inner.Copy(sourceFileName, destFileName, overwrite);

        FileSystemStream IFile.Create(string path) => inner.Create(path);

        public FileSystemStream Create(string path, int bufferSize) => inner.Create(path, bufferSize);

        public FileSystemStream Create(string path, int bufferSize, FileOptions options) => inner.Create(path, bufferSize, options);

        public IFileSystemInfo CreateSymbolicLink(string path, string pathToTarget) => inner.CreateSymbolicLink(path, pathToTarget);
        [SupportedOSPlatform("windows")]
        public void Decrypt(string path) => inner.Decrypt(path);
        [SupportedOSPlatform("windows")]
        public void Encrypt(string path) => inner.Encrypt(path);

        public FileAttributes GetAttributes(SafeFileHandle fileHandle) => inner.GetAttributes(fileHandle);

        public DateTime GetCreationTime(SafeFileHandle fileHandle) => inner.GetCreationTime(fileHandle);

        public DateTime GetCreationTimeUtc(string path) => inner.GetCreationTimeUtc(path);

        public DateTime GetCreationTimeUtc(SafeFileHandle fileHandle) => inner.GetCreationTimeUtc(fileHandle);

        public DateTime GetLastAccessTime(SafeFileHandle fileHandle) => inner.GetLastAccessTime(fileHandle);

        public DateTime GetLastAccessTimeUtc(string path) => inner.GetLastAccessTimeUtc(path);

        public DateTime GetLastAccessTimeUtc(SafeFileHandle fileHandle) => inner.GetLastAccessTimeUtc(fileHandle);

        public DateTime GetLastWriteTime(SafeFileHandle fileHandle) => inner.GetLastWriteTime(fileHandle);

        public DateTime GetLastWriteTimeUtc(string path) => inner.GetLastWriteTimeUtc(path);

        public DateTime GetLastWriteTimeUtc(SafeFileHandle fileHandle) => inner.GetLastWriteTimeUtc(fileHandle);
        [UnsupportedOSPlatform("windows")]
        public UnixFileMode GetUnixFileMode(string path) => inner.GetUnixFileMode(path);
        [UnsupportedOSPlatform("windows")]
        public UnixFileMode GetUnixFileMode(SafeFileHandle fileHandle) => inner.GetUnixFileMode(fileHandle);

        public void Move(string sourceFileName, string destFileName, bool overwrite) => inner.Move(sourceFileName, destFileName, overwrite);

        FileSystemStream IFile.Open(string path, FileMode mode) => inner.Open(path, mode);

        public FileSystemStream Open(string path, FileMode mode, FileAccess access) => inner.Open(path, mode, access);

        public FileSystemStream Open(string path, FileMode mode, FileAccess access, FileShare share) => inner.Open(path, mode, access, share);

        public FileSystemStream Open(string path, FileStreamOptions options) => inner.Open(path, options);

        FileSystemStream IFile.OpenRead(string path) => inner.OpenRead(path);

        FileSystemStream IFile.OpenWrite(string path) => inner.OpenWrite(path);

        public string[] ReadAllLines(string path, Encoding encoding) => inner.ReadAllLines(path, encoding);

        public Task<string[]> ReadAllLinesAsync(string path, CancellationToken cancellationToken = default) => inner.ReadAllLinesAsync(path, cancellationToken);

        public Task<string[]> ReadAllLinesAsync(string path, Encoding encoding, CancellationToken cancellationToken = default) => inner.ReadAllLinesAsync(path, encoding, cancellationToken);

        public Task<string> ReadAllTextAsync(string path, Encoding encoding, CancellationToken cancellationToken = default) => inner.ReadAllTextAsync(path, encoding, cancellationToken);

        public IEnumerable<string> ReadLines(string path, Encoding encoding) => inner.ReadLines(path, encoding);

        public IAsyncEnumerable<string> ReadLinesAsync(string path, CancellationToken cancellationToken = default) => inner.ReadLinesAsync(path, cancellationToken);

        public IAsyncEnumerable<string> ReadLinesAsync(string path, Encoding encoding, CancellationToken cancellationToken = default) => inner.ReadLinesAsync(path, encoding, cancellationToken);

        public void Replace(string sourceFileName, string destinationFileName, string? destinationBackupFileName) => inner.Replace(sourceFileName, destinationFileName, destinationBackupFileName);

        public void Replace(string sourceFileName, string destinationFileName, string? destinationBackupFileName, bool ignoreMetadataErrors) => inner.Replace(sourceFileName, destinationFileName, destinationBackupFileName, ignoreMetadataErrors);

        public IFileSystemInfo? ResolveLinkTarget(string linkPath, bool returnFinalTarget) => inner.ResolveLinkTarget(linkPath, returnFinalTarget);

        public void SetAttributes(SafeFileHandle fileHandle, FileAttributes fileAttributes) => inner.SetAttributes(fileHandle, fileAttributes);

        public void SetCreationTime(string path, DateTime creationTime) => inner.SetCreationTime(path, creationTime);

        public void SetCreationTime(SafeFileHandle fileHandle, DateTime creationTime) => inner.SetCreationTime(fileHandle, creationTime);

        public void SetCreationTimeUtc(string path, DateTime creationTimeUtc) => inner.SetCreationTimeUtc(path, creationTimeUtc);

        public void SetCreationTimeUtc(SafeFileHandle fileHandle, DateTime creationTimeUtc) => inner.SetCreationTimeUtc(fileHandle, creationTimeUtc);

        public void SetLastAccessTime(string path, DateTime lastAccessTime) => inner.SetLastAccessTime(path, lastAccessTime);

        public void SetLastAccessTime(SafeFileHandle fileHandle, DateTime lastAccessTime) => inner.SetLastAccessTime(fileHandle, lastAccessTime);

        public void SetLastAccessTimeUtc(string path, DateTime lastAccessTimeUtc) => inner.SetLastAccessTimeUtc(path, lastAccessTimeUtc);

        public void SetLastAccessTimeUtc(SafeFileHandle fileHandle, DateTime lastAccessTimeUtc) => inner.SetLastAccessTimeUtc(fileHandle, lastAccessTimeUtc);

        public void SetLastWriteTime(string path, DateTime lastWriteTime) => inner.SetLastWriteTime(path, lastWriteTime);

        public void SetLastWriteTime(SafeFileHandle fileHandle, DateTime lastWriteTime) => inner.SetLastWriteTime(fileHandle, lastWriteTime);

        public void SetLastWriteTimeUtc(string path, DateTime lastWriteTimeUtc) => inner.SetLastWriteTimeUtc(path, lastWriteTimeUtc);

        public void SetLastWriteTimeUtc(SafeFileHandle fileHandle, DateTime lastWriteTimeUtc) => inner.SetLastWriteTimeUtc(fileHandle, lastWriteTimeUtc);
        [UnsupportedOSPlatform("windows")]
        public void SetUnixFileMode(string path, UnixFileMode mode) => inner.SetUnixFileMode(path, mode);
        [UnsupportedOSPlatform("windows")]
        public void SetUnixFileMode(SafeFileHandle fileHandle, UnixFileMode mode) => inner.SetUnixFileMode(fileHandle, mode);

        public void WriteAllBytes(string path, ReadOnlySpan<byte> bytes) => inner.WriteAllBytes(path, bytes);

        public Task WriteAllBytesAsync(string path, ReadOnlyMemory<byte> bytes, CancellationToken cancellationToken = default) => inner.WriteAllBytesAsync(path, bytes, cancellationToken);

        public void WriteAllLines(string path, string[] contents) => inner.WriteAllLines(path, contents);

        public void WriteAllLines(string path, string[] contents, Encoding encoding) => inner.WriteAllLines(path, contents, encoding);

        public void WriteAllLines(string path, IEnumerable<string> contents, Encoding encoding) => inner.WriteAllLines(path, contents, encoding);

        public Task WriteAllLinesAsync(string path, IEnumerable<string> contents, CancellationToken cancellationToken = default) => inner.WriteAllLinesAsync(path, contents, cancellationToken);

        public Task WriteAllLinesAsync(string path, IEnumerable<string> contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.WriteAllLinesAsync(path, contents, encoding, cancellationToken);

        public void WriteAllText(string path, ReadOnlySpan<char> contents) => inner.WriteAllText(path, contents);

        public void WriteAllText(string path, ReadOnlySpan<char> contents, Encoding encoding) => inner.WriteAllText(path, contents, encoding);

        public Task WriteAllTextAsync(string path, string? contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.WriteAllTextAsync(path, contents, encoding, cancellationToken);

        public Task WriteAllTextAsync(string path, ReadOnlyMemory<char> contents, CancellationToken cancellationToken = default) => inner.WriteAllTextAsync(path, contents, cancellationToken);

        public Task WriteAllTextAsync(string path, ReadOnlyMemory<char> contents, Encoding encoding, CancellationToken cancellationToken = default) => inner.WriteAllTextAsync(path, contents, encoding, cancellationToken);
    }

    // Gated IDirectory

    private sealed class GatedDirectory(IDirectory inner, Action<string> gate) : IDirectory
    {
        public IFileSystem FileSystem => inner.FileSystem;

        public IDirectoryInfo CreateDirectory(string path) { gate(path); return inner.CreateDirectory(path); }
        public void Delete(string path) { gate(path); inner.Delete(path); }
        public void Delete(string path, bool recursive) { gate(path); inner.Delete(path, recursive); }
        public bool Exists(string? path) => inner.Exists(path);
        public IEnumerable<string> EnumerateFiles(string path) { gate(path); return inner.EnumerateFiles(path); }
        public IEnumerable<string> EnumerateDirectories(string path) { gate(path); return inner.EnumerateDirectories(path); }
        public string[] GetFiles(string path) { gate(path); return inner.GetFiles(path); }
        public string[] GetDirectories(string path) { gate(path); return inner.GetDirectories(path); }
        public void Move(string sourceDirName, string destDirName) { gate(sourceDirName); gate(destDirName); inner.Move(sourceDirName, destDirName); }
        public string GetCurrentDirectory() => inner.GetCurrentDirectory();
        public void SetCurrentDirectory(string path) => inner.SetCurrentDirectory(path);
        public string[] GetLogicalDrives() => inner.GetLogicalDrives();
        public IDirectoryInfo? GetParent(string path) => inner.GetParent(path);
        public IEnumerable<string> EnumerateFileSystemEntries(string path) { gate(path); return inner.EnumerateFileSystemEntries(path); }
        public string[] GetFileSystemEntries(string path) { gate(path); return inner.GetFileSystemEntries(path); }

        public IDirectoryInfo CreateDirectory(string path, UnixFileMode unixCreateMode) => inner.CreateDirectory(path, unixCreateMode);

        public IFileSystemInfo CreateSymbolicLink(string path, string pathToTarget) => inner.CreateSymbolicLink(path, pathToTarget);

        public IDirectoryInfo CreateTempSubdirectory(string? prefix = null) => inner.CreateTempSubdirectory(prefix);

        public IEnumerable<string> EnumerateDirectories(string path, string searchPattern) => inner.EnumerateDirectories(path, searchPattern);

        public IEnumerable<string> EnumerateDirectories(string path, string searchPattern, SearchOption searchOption) => inner.EnumerateDirectories(path, searchPattern, searchOption);

        public IEnumerable<string> EnumerateDirectories(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.EnumerateDirectories(path, searchPattern, enumerationOptions);

        public IEnumerable<string> EnumerateFiles(string path, string searchPattern) => inner.EnumerateFiles(path, searchPattern);

        public IEnumerable<string> EnumerateFiles(string path, string searchPattern, SearchOption searchOption) => inner.EnumerateFiles(path, searchPattern, searchOption);

        public IEnumerable<string> EnumerateFiles(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.EnumerateFiles(path, searchPattern, enumerationOptions);

        public IEnumerable<string> EnumerateFileSystemEntries(string path, string searchPattern) => inner.EnumerateFileSystemEntries(path, searchPattern);

        public IEnumerable<string> EnumerateFileSystemEntries(string path, string searchPattern, SearchOption searchOption) => inner.EnumerateFileSystemEntries(path, searchPattern, searchOption);

        public IEnumerable<string> EnumerateFileSystemEntries(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.EnumerateFileSystemEntries(path, searchPattern, enumerationOptions);

        public DateTime GetCreationTime(string path) => inner.GetCreationTime(path);

        public DateTime GetCreationTimeUtc(string path) => inner.GetCreationTimeUtc(path);

        public string[] GetDirectories(string path, string searchPattern) => inner.GetDirectories(path, searchPattern);

        public string[] GetDirectories(string path, string searchPattern, SearchOption searchOption) => inner.GetDirectories(path, searchPattern, searchOption);

        public string[] GetDirectories(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.GetDirectories(path, searchPattern, enumerationOptions);

        public string GetDirectoryRoot(string path) => inner.GetDirectoryRoot(path);

        public string[] GetFiles(string path, string searchPattern) => inner.GetFiles(path, searchPattern);

        public string[] GetFiles(string path, string searchPattern, SearchOption searchOption) => inner.GetFiles(path, searchPattern, searchOption);

        public string[] GetFiles(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.GetFiles(path, searchPattern, enumerationOptions);

        public string[] GetFileSystemEntries(string path, string searchPattern) => inner.GetFileSystemEntries(path, searchPattern);

        public string[] GetFileSystemEntries(string path, string searchPattern, SearchOption searchOption) => inner.GetFileSystemEntries(path, searchPattern, searchOption);

        public string[] GetFileSystemEntries(string path, string searchPattern, EnumerationOptions enumerationOptions) => inner.GetFileSystemEntries(path, searchPattern, enumerationOptions);

        public DateTime GetLastAccessTime(string path) => inner.GetLastAccessTime(path);

        public DateTime GetLastAccessTimeUtc(string path) => inner.GetLastAccessTimeUtc(path);

        public DateTime GetLastWriteTime(string path) => inner.GetLastWriteTime(path);

        public DateTime GetLastWriteTimeUtc(string path) => inner.GetLastWriteTimeUtc(path);

        public IFileSystemInfo? ResolveLinkTarget(string linkPath, bool returnFinalTarget) => inner.ResolveLinkTarget(linkPath, returnFinalTarget);

        public void SetCreationTime(string path, DateTime creationTime) => inner.SetCreationTime(path, creationTime);

        public void SetCreationTimeUtc(string path, DateTime creationTimeUtc) => inner.SetCreationTimeUtc(path, creationTimeUtc);

        public void SetLastAccessTime(string path, DateTime lastAccessTime) => inner.SetLastAccessTime(path, lastAccessTime);

        public void SetLastAccessTimeUtc(string path, DateTime lastAccessTimeUtc) => inner.SetLastAccessTimeUtc(path, lastAccessTimeUtc);

        public void SetLastWriteTime(string path, DateTime lastWriteTime) => inner.SetLastWriteTime(path, lastWriteTime);

        public void SetLastWriteTimeUtc(string path, DateTime lastWriteTimeUtc) => inner.SetLastWriteTimeUtc(path, lastWriteTimeUtc);
    }
}
