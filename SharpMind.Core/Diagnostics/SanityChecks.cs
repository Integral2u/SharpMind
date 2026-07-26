using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;

namespace SharpMind.Core.Diagnostics;

public static class SanityChecks
{
    [Conditional("DEBUG")]
    public static void That(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException($"Sanity check failed: {message}");
    }

    [Conditional("DEBUG")]
    public static void NotNull([NotNull] object? obj, string name)
    {
        if (obj is null) throw new ArgumentNullException(name, $"Sanity check failed: {name} is null");
    }

    [Conditional("DEBUG")]
    public static void InRange<T>(T value, T min, T max, string name) where T : INumber<T>
    {
        if (value < min || value > max)
            throw new ArgumentOutOfRangeException(name, value, $"Sanity check failed: {name} = {value} not in [{min}..{max}]");
    }

    [Conditional("DEBUG")]
    public static void Equal<T>(T expected, T actual, string message) where T : IEquatable<T>
    {
        if (!expected.Equals(actual))
            throw new InvalidOperationException($"Sanity check failed: {message}. Expected {expected}, got {actual}.");
    }

    [Conditional("DEBUG")]
    public static void SameDimensions(ReadOnlySpan<int> a, ReadOnlySpan<int> b, string nameA, string nameB)
    {
        if (!a.SequenceEqual(b))
            throw new InvalidOperationException($"Sanity check failed: {nameA} shape [{string.Join(",", a.ToArray())}] != {nameB} shape [{string.Join(",", b.ToArray())}]");
    }

    [Conditional("DEBUG")]
    public static void NotNaN(ReadOnlySpan<float> values, string label)
    {
        for (int i = 0; i < values.Length; i++)
            if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
                throw new InvalidOperationException($"Sanity check failed: {label}[{i}] = {values[i]}");
    }
}
