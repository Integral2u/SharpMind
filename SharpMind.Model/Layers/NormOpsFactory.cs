using JigSawDotNet;

namespace SharpMind.Model.Layers;

public static class NormOpsFactory
{
    public static NormOps Create(SharpMindConfig config)
        => Assembler.CreateInstance<NormOps>(config.ToJigSawMapping());

    public static NormOps SetDefault(SharpMindConfig config)
    {
        var ops = Create(config);
        NormOps.SetDefault(ops);
        return ops;
    }
}
