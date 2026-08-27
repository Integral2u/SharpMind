using System.IO.MemoryMappedFiles;

namespace SharpMind.Model.Format;

/// <summary>
/// Opens a seekable, readable stream over a model file's tensor data, for
/// <see cref="GgufLoader"/> and <see cref="SmmLoader"/>. Both loaders'
/// metadata/tokenizer reads already go through plain <c>File.OpenRead</c>
/// (see <c>LoadMeta</c>) and need no changes -- this factory exists only
/// for the tensor-data reads, which used <see cref="MemoryMappedFile"/>
/// directly.
///
/// By default (<paramref name="useSafeIo"/> = false), behavior is byte-for-
/// byte what it always was: a memory-mapped view, letting the OS page large
/// files in on demand rather than buffering them, which matters for models
/// too big to comfortably hold in RAM twice.
///
/// Set <paramref name="useSafeIo"/> = true on platforms where
/// <see cref="MemoryMappedFile"/> isn't available at all -- notably
/// wasm-browser, where <c>MemoryMappedFile.CreateFromFile</c> throws
/// <see cref="PlatformNotSupportedException"/> unconditionally -- to fall
/// back to a plain <see cref="FileStream"/> instead. On WASM this works
/// transparently against Blazor's virtual (in-memory) filesystem once the
/// fetched model bytes have been written to that path, using the exact same
/// path string passed to the loader's constructor; no separate byte-buffer
/// plumbing is needed.
/// </summary>
internal static class WeightStreamFactory
{
    public static Stream Open(string path, bool useSafeIo)
    {
        if (useSafeIo)
            return File.OpenRead(path);

        var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        return new MappedViewStream(mmf, mmf.CreateViewStream(0, 0, MemoryMappedFileAccess.Read));
    }

    /// <summary>
    /// Couples a MemoryMappedFile's lifetime to its view stream's. Disposing
    /// the view stream alone does not necessarily release the underlying
    /// mapping -- callers that only ever see this as a plain Stream (every
    /// call site in both loaders) now get correct disposal of both for free
    /// from a single `using`, instead of needing to keep a separate `using
    /// var mmf` alive alongside it as the original inline call sites did.
    /// </summary>
    private sealed class MappedViewStream(MemoryMappedFile mmf, MemoryMappedViewStream view) : Stream
    {
        public override bool CanRead => view.CanRead;
        public override bool CanSeek => view.CanSeek;
        public override bool CanWrite => false;
        public override long Length => view.Length;
        public override long Position { get => view.Position; set => view.Position = value; }
        public override void Flush() => view.Flush();
        public override int Read(byte[] buffer, int offset, int count) => view.Read(buffer, offset, count);
        public override long Seek(long offset, SeekOrigin origin) => view.Seek(offset, origin);
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                view.Dispose();
                mmf.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
