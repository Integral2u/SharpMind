using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpMind.Core.Diagnostics;

public static class SanityChecks
{
    /// <summary>
    /// Debug-only logging that is completely elided from Release builds.
    /// Use in catch blocks and silent-failure paths where a debugger attach
    /// is impractical but the information is useless to end users.
    /// </summary>
    [Conditional("DEBUG")]
    public static void WriteLine(string message) => Debug.WriteLine(message);

    [Conditional("DEBUG")]
    public static void That(bool condition, string message,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0)
    {
        if (!condition) throw new InvalidOperationException($"Sanity check failed ({caller} at {file}:{line}): {message}");
    }

    [Conditional("DEBUG")]
    public static void NotNull([NotNull] object? obj, string name,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0)
    {
        if (obj is null) throw new ArgumentNullException(name, $"Sanity check failed ({caller} at {file}:{line}): {name} is null");
    }

    [Conditional("DEBUG")]
    public static void InRange<T>(T value, T min, T max, string name,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) where T : INumber<T>
    {
        if (value < min || value > max)
            throw new ArgumentOutOfRangeException(name, value, $"Sanity check failed ({caller} at {file}:{line}): {name} = {value} not in [{min}..{max}]");
    }

    /// <summary>Fast bounds check for flat indices: <c>0 &lt;= index &lt; length</c>.</summary>
    [Conditional("DEBUG")]
    public static void IndexInRange(int index, int length, string name,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0)
    {
        if ((uint)index >= (uint)length)
            throw new ArgumentOutOfRangeException(name, index, $"Sanity check failed ({caller} at {file}:{line}): {name} = {index} not in [0..{length - 1}]");
    }

    [Conditional("DEBUG")]
    public static void Equal<T>(T expected, T actual, string message,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0) where T : IEquatable<T>
    {
        if (!expected.Equals(actual))
            throw new InvalidOperationException($"Sanity check failed ({caller} at {file}:{line}): {message}. Expected {expected}, got {actual}.");
    }

    [Conditional("DEBUG")]
    public static void SameDimensions(ReadOnlySpan<int> a, ReadOnlySpan<int> b, string nameA, string nameB,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0)
    {
        if (!a.SequenceEqual(b))
            throw new InvalidOperationException($"Sanity check failed ({caller} at {file}:{line}): {nameA} shape [{string.Join(",", a.ToArray())}] != {nameB} shape [{string.Join(",", b.ToArray())}]");
    }

    [Conditional("DEBUG")]
    public static void NotNaN(ReadOnlySpan<float> values, string label,
        [CallerMemberName] string caller = "",
        [CallerFilePath] string file = "",
        [CallerLineNumber] int line = 0)
    {
        for (int i = 0; i < values.Length; i++)
            if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                throw new InvalidOperationException($"Sanity check failed ({caller} at {file}:{line}): {label}[{i}] = {values[i]}");
    }
}
