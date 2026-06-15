using System.Numerics;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Numerics;

/// <summary>
/// Tensor initialisation strategies. All methods return a new tensor;
/// none modify an existing one (makes it easy to reason about weight init).
/// </summary>
public static class TensorInit
{
    // uniform

    /// <summary>Fills with U[low, high) samples.</summary>
    public static Tensor<T> Uniform<T>(TensorShape shape, T low, T high, int? seed = null)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var rng    = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var tensor = new Tensor<T>(shape);
        T   range  = high - low;
        for (int i = 0; i < tensor.ElementCount; i++)
            tensor[i] = T.CreateChecked(rng.NextDouble()) * range + low;
        return tensor;
    }

    // normal

    /// <summary>Fills with N(mean, std²) samples via Box-Muller.</summary>
    public static Tensor<T> Normal<T>(
        TensorShape shape, T mean, T std, int? seed = null)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        var rng    = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var tensor = new Tensor<T>(shape);

        for (int i = 0; i < tensor.ElementCount; i++)
            tensor[i] = T.CreateChecked(SampleNormal(rng)) * std + mean;

        return tensor;
    }

    // Xavier / Glorot uniform

    /// <summary>
    /// Xavier uniform initialisation (Glorot &amp; Bengio, 2010).
    /// Recommended for tanh / linear layers.
    /// U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
    /// </summary>
    public static Tensor<T> XavierUniform<T>(
        TensorShape shape, int? seed = null)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        (int fanIn, int fanOut) = FanInOut(shape);
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        T lim = T.CreateChecked(limit);
        return Uniform(shape, T.Zero - lim, lim, seed);
    }

    // He / Kaiming normal

    /// <summary>
    /// He / Kaiming normal initialisation (He et al., 2015).
    /// Recommended for ReLU / GELU layers.
    /// N(0, sqrt(2 / fan_in))
    /// </summary>
    public static Tensor<T> HeNormal<T>(
        TensorShape shape, int? seed = null)
        where T : unmanaged, INumber<T>, IFloatingPoint<T>
    {
        (int fanIn, _) = FanInOut(shape);
        T std = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        return Normal(shape, T.Zero, std, seed);
    }

    // helpers

    /// <summary>
    /// Estimates fan-in and fan-out from a weight tensor's shape.
    /// Handles 1-D (bias), 2-D (linear), and 4-D (conv) tensors.
    /// </summary>
    private static (int fanIn, int fanOut) FanInOut(TensorShape shape)
    {
        return shape.Rank switch
        {
            1 => (shape[0], shape[0]),
            2 => (shape[0], shape[1]), // [in_features, out_features]
            _ => throw new NotSupportedException(
                $"Fan-in/out heuristic not defined for rank-{shape.Rank} tensors.")
        };
    }

    /// <summary>
    /// Box-Muller transform: converts two U(0,1) samples to one N(0,1) sample.
    /// </summary>
    private static double SampleNormal(Random rng)
    {
        // Avoid log(0)
        double u1 = Math.Max(rng.NextDouble(), 1e-10);
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
