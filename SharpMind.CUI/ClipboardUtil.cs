using System.Runtime.InteropServices;

namespace SharpMind.CUI;

/// <summary>
/// Minimal Win32 clipboard writer. The CUI is a Windows console app and
/// Terminal.Gui 1.19 only surfaces <see cref="Terminal.Gui.IClipboard"/> with
/// no static implementation, so the user32 path is the dependable way to get
/// error text out of the terminal into a paste target. Retries opening the
/// clipboard a few times because another process may be holding it open.
/// </summary>
internal static class ClipboardUtil
{
    private const uint CfUnicodeText = 13;
    private const uint GmemMoveable = 0x0002;

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool OpenClipboard(IntPtr hWndNewOwner);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool EmptyClipboard();

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr SetClipboardData(uint uFormat, IntPtr hMem);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool CloseClipboard();

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GlobalAlloc(uint uFlags, nuint dwBytes);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GlobalLock(IntPtr hMem);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GlobalUnlock(IntPtr hMem);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GlobalFree(IntPtr hMem);

    /// <summary>Writes <paramref name="text"/> to the clipboard (best effort, with retry).</summary>
    public static bool CopyToClipboard(string text)
    {
        if (string.IsNullOrEmpty(text)) return false;

        for (int attempt = 0; attempt < 3; attempt++)
        {
            if (TrySet(text)) return true;
            if (attempt < 2) System.Threading.Thread.Sleep(50);
        }
        return false;
    }

    private static bool TrySet(string text)
    {
        if (!OpenClipboard(IntPtr.Zero)) return false;
        try
        {
            if (!EmptyClipboard()) return false;

            int chars = text.Length;
            int bytes = (chars + 1) * sizeof(char);
            IntPtr hGlobal = GlobalAlloc(GmemMoveable, (nuint)bytes);
            if (hGlobal == IntPtr.Zero) return false;

            IntPtr dest = GlobalLock(hGlobal);
            bool locked = dest != IntPtr.Zero;
            if (locked)
            {
                Marshal.Copy(text.ToCharArray(), 0, dest, chars);
                Marshal.WriteInt16(dest, chars * 2, 0);
            }
            if (hGlobal != IntPtr.Zero) GlobalUnlock(hGlobal);

            if (!locked)
            {
                GlobalFree(hGlobal);
                return false;
            }

            // On success the clipboard owns the memory — do not free hGlobal.
            if (SetClipboardData(CfUnicodeText, hGlobal) != IntPtr.Zero)
            {
                hGlobal = IntPtr.Zero;
                return true;
            }

            if (hGlobal != IntPtr.Zero) GlobalFree(hGlobal);
            return false;
        }
        finally
        {
            CloseClipboard();
        }
    }
}