namespace SharpMind.Model.Format;

/// <summary>
/// Supported data types for model weights and tensors.
/// Maps to common formats: FP32, FP16, BF16, INT8, INT4.
/// </summary>
public enum Dtype
{
    /// <summary>32-bit float (standard training precision)</summary>
    F32 = 0,
    
    /// <summary>16-bit float (half precision)</summary>
    F16 = 1,
    
    /// <summary>Brain float16 (better dynamic range for ML)</summary>
    BF16 = 2,
    
    /// <summary>8-bit signed integer (symmetric quantization)</summary>
    INT8 = 3,
    
    /// <summary>4-bit packed integer (high compression)</summary>
    INT4 = 4,
}

/// <summary>
/// Quantization configuration for model weights.
/// </summary>
public sealed record QuantConfig
{
    public required Dtype Dtype { get; init; }
    public int BlockSize { get; init; } = 64;
    public bool Symmetric { get; init; } = true;
    
    public static QuantConfig None => new() { Dtype = Dtype.F32 };
    public static QuantConfig Int8 => new() { Dtype = Dtype.INT8, BlockSize = 128, Symmetric = true };
    public static QuantConfig Int4 => new() { Dtype = Dtype.INT4, BlockSize = 64, Symmetric = false };
    public static QuantConfig FP16 => new() { Dtype = Dtype.F16 };
    public static QuantConfig BF16 => new() { Dtype = Dtype.BF16 };
}