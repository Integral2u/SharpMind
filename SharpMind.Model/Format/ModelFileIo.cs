using System.Runtime.InteropServices;

namespace SharpMind.Model.Format;

/// <summary>
/// Model-file opens that tolerate a transient sharing violation. Antivirus
/// and indexers on Windows can briefly hold a freshly written .gguf/.smm
/// open with an exclusive handle; a load triggered right after an export
/// (or a second model opened from the same path while the first is
/// streaming) must not fail the whole app for that. Retries only the two
/// Windows "someone else has this file" errors — a genuine missing file
/// or a read fault still throws immediately.
/// </summary>
internal static class ModelFileIo
{
    public const int MaxRetries = 8;

    private const int ErrorSharingViolation = 32;
    private const int ErrorLockViolation = 33;
    private const int BaseRetryDelayMs = 50;

    /// <summary>Opens the file read-only, retrying a transient share/lock violation.</summary>
    public static FileStream OpenRead(string path) => Open(Path.GetFullPath(path), () => File.OpenRead(path));

    /// <summary>Opens the model's tensor-data stream, retrying a transient share/lock violation.</summary>
    public static Stream OpenModelStream(string path, bool useSafeIo) =>
        Open(Path.GetFullPath(path), () => WeightStreamFactory.Open(path, useSafeIo));

    private static T Open<T>(string fullPath, Func<T> open) where T : Stream
    {
        for (int attempt = 0; ; attempt++)
        {
            try
            {
                return open();
            }
            catch (IOException ex) when (attempt < MaxRetries && IsTransient(ex))
            {
                Thread.Sleep(BaseRetryDelayMs * (attempt + 1));
            }
            catch (IOException) when (attempt < MaxRetries)
            {
                throw;
            }
            catch (IOException ex)
            {
                throw new IOException(
                    $"Could not open model file '{fullPath}' after {MaxRetries + 1} attempts: {ex.Message}", ex);
            }
        }
    }

    private static bool IsTransient(IOException ex)
    {
        int code = Marshal.GetHRForException(ex) & 0xFFFF;
        return code == ErrorSharingViolation || code == ErrorLockViolation;
    }
}