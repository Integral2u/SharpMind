namespace SharpMind.Model.Format;

/// <summary>
/// Shared helpers for computing element counts and checking materializability
/// of tensors during model loading. Used by both GgufLoader and SmmLoader.
/// </summary>
internal static class TensorLoadHelper
{
    /// <summary>
    /// Computes the total element count for a tensor shape as a long.
    /// Use this instead of int arithmetic to avoid overflow on large shapes.
    /// </summary>
    public static long ComputeElementCount(int[] shape)
    {
        long count = 1;
        foreach (int d in shape) count *= d;
        return count;
    }

    /// <summary>
    /// Computes element count and throws if it exceeds <see cref="int.MaxValue"/>.
    /// Returns the count as an int for use with ArrayPool-sized buffers.
    /// </summary>
    public static int ComputeElementCountChecked(int[] shape)
    {
        long count = ComputeElementCount(shape);
        if (count > int.MaxValue)
            throw new NotSupportedException(
                $"Tensor with shape [{string.Join(",", shape)}] has {count:N0} elements, " +
                $"more than a single float buffer can hold (max {int.MaxValue:N0}).");
        return (int)count;
    }

    /// <summary>
    /// Guards a long→int cast for raw byte sizes. Throws if the size
    /// exceeds what can be passed to APIs expecting int-length buffers.
    /// </summary>
    public static int CheckedInt(long value, string label)
    {
        if (value > int.MaxValue)
            throw new NotSupportedException(
                $"{label} is {value:N0}, exceeding int.MaxValue ({int.MaxValue:N0}).");
        return (int)value;
    }
}
