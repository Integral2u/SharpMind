using System.Diagnostics;

namespace SharpMind.Core;

/// <summary>
/// Debug-only logging that is completely elided from Release builds.
/// Use in catch blocks and silent-failure paths where a debugger attach
/// is impractical but the information is useless to end users.
/// </summary>
public static class InternalLog
{
    [Conditional("DEBUG")]
    public static void WriteLine(string message) => Debug.WriteLine(message);
}
