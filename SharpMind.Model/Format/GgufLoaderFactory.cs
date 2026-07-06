using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

public static class GgufLoaderFactory
{
    private static GgufLoader? _default;

    public static GgufLoader Default => _default ??= Create();

    public static GgufLoader Create(QuantizationConfig? config = null)
    {
        var qOps = QuantizationFactory.Create(config?.Hardware ?? HardwareTier.Auto);
        return new GgufLoader(qOps);
    }

    public static GgufLoader Create(QuantizationOps qOps)
    {
        return new GgufLoader(qOps);
    }
}
