using JigSawDotNet;

namespace SharpMind.Core.Ops;

/// <summary>
/// Creates <see cref="TensorOps"/> instances wired by JigSawDotNet.
/// </summary>
public static class TensorOpsFactory
{
    /// <summary>
    /// Assembles and returns a <see cref="TensorOps"/> for the given config.
    /// </summary>
    public static TensorOps Create(SharpMindConfig config)
    {
        var mapping = config.ToJigSawMapping();
        return Assembler.CreateInstance<TensorOps>(mapping);
    }

    /// <summary>
    /// Assembles a <see cref="TensorOps"/>, sets it as <see cref="TensorOps.Default"/>,
    /// and returns it. Call once at application startup.
    /// </summary>
    public static TensorOps SetDefault(SharpMindConfig config)
    {
        var ops = Create(config);
        TensorOps.SetDefault(ops);
        return ops;
    }
}
