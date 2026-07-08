using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

public readonly struct KvPair { public required string Key { get; init; } public required object Value { get; init; } }

public readonly struct TensorInfo { public required string Name { get; init; } public required QuantDType Dtype { get; init; } public required int[] Shape { get; init; } public required long Offset { get; init; } }
