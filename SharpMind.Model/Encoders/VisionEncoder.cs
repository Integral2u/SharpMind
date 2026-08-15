using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Encoders;

/// <summary>
/// Vision encoder: turns a raw image into a sequence of HiddenDim patch
/// embeddings that a decoder transformer can attend to.
///
/// Pipeline (mirrors ViT-style patch embedding):
///   image [B, Channels, ImageSize, ImageSize]
///     → patches [B, NumPatches, patchDim]         (patchDim = Channels × Patch²)
///     → linear projection → [B, NumPatches, HiddenDim]
///     → learned positional embedding + RMS norm
///     → output [B, NumPatches, HiddenDim]
///
/// The encoder owns its projection, position embedding and norm parameters,
/// exposed through <see cref="Parameters"/> for a future optimizer.
/// </summary>
public sealed class VisionEncoder : IDisposable
{
    private readonly int _patchSize;
    private readonly int _channels;
    private readonly int _imageSize;
    private readonly int _patchDim;
    private readonly int _numPatches;
    private readonly LinearLayer _proj;
    private readonly Tensor<float> _posEmbed;   // [NumPatches, HiddenDim]
    private readonly Tensor<float> _normWeight; // [HiddenDim]
    private readonly int _hiddenDim;
    private readonly float _eps;
    private bool _disposed;

    public VisionEncoder(ModelConfig config, Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!config.HasVision)
            throw new ArgumentException("Vision encoder requires VisionPatchSize.", nameof(config));
        config.Validate();

        rng ??= Random.Shared;
        _patchSize = config.VisionPatchSize!.Value;
        _channels = config.VisionChannels;
        _imageSize = config.VisionImageSize;
        _patchDim = config.VisionPatchDim;
        _numPatches = config.VisionNumPatches;
        _hiddenDim = config.HiddenDim;
        _eps = config.NormEps;

        _proj = LinearLayerFactory.Create(
            "vision.patch_proj", _patchDim, _hiddenDim, bias: false,
            weight: null, biasTensor: null, QuantDType.F32);

        // Learned position embeddings, small random init.
        _posEmbed = new Tensor<float>(_numPatches, _hiddenDim);
        FillUniformSmall(_posEmbed.Data, rng);

        // Norm weight starts at one (identity).
        _normWeight = Tensor<float>.Ones(_hiddenDim);
    }

    /// <summary>Flattened size of a single image patch (channels × patch²).</summary>
    public int InFeatures => _patchDim;
    /// <summary>Number of patch tokens produced for one image.</summary>
    public int NumTokens => _numPatches;
    /// <summary>Hidden (transformer) dimension of the emitted embeddings.</summary>
    public int HiddenDim => _hiddenDim;

    public void SetNormWeight(ReadOnlySpan<float> data)
    {
        ThrowIfDisposed();
        if (data.Length != _hiddenDim)
            throw new ArgumentException($"Expected {_hiddenDim} norm weights, got {data.Length}.");
        data.CopyTo(_normWeight.Data);
    }

    public Tensor<float>? GetPosEmbed() => _disposed ? null : _posEmbed;

    /// <summary>
    /// Embeds a raw image into patch tokens.
    /// Input:  [Batch, Channels, ImageSize, ImageSize]
    /// Output: [Batch, NumPatches, HiddenDim]
    /// </summary>
    public Tensor<float> Forward(Tensor<float> image)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(image);
        if (image.Rank != 4)
            throw new ArgumentException($"Vision encoder expects a 4D image [B, C, H, W], got rank {image.Rank}.");
        if (image.Shape[1] != _channels)
            throw new ArgumentException($"Vision encoder expects {_channels} channels, got {image.Shape[1]}.");
        if (image.Shape[2] != _imageSize || image.Shape[3] != _imageSize)
            throw new ArgumentException(
                $"Vision encoder expects a {_imageSize}×{_imageSize} image, got {image.Shape[2]}×{image.Shape[3]}.");

        int batch = image.Shape[0];

        // Patchify → [B*NumPatches, patchDim]
        using var patches = new Tensor<float>(batch * _numPatches, _patchDim);
        Patchify(image, patches.Data, batch);

        // Linear projection → [B*NumPatches, HiddenDim], then split back to [B, NumPatches, H]
        using var projected2d = _proj.Forward(patches);
        using var projected = projected2d.Reshape(batch, _numPatches, _hiddenDim);

        // Add positional embedding [NumPatches, H] to every batch row.
        var result = new Tensor<float>(batch, _numPatches, _hiddenDim);
        var resultData = result.Data;
        var posData = _posEmbed.Data;
        int hidden = _hiddenDim;
        projected.Data.CopyTo(resultData);
        for (int b = 0; b < batch; b++)
        {
            int rowBase = b * _numPatches * hidden;
            for (int p = 0; p < _numPatches; p++)
            {
                int rowOff = rowBase + p * hidden;
                for (int h = 0; h < hidden; h++)
                    resultData[rowOff + h] += posData[p * hidden + h];
            }
        }

        // RMS normalise each row (self-contained, no global NormOps dependency).
        ApplyRmsNorm(result, _normWeight.Data, _eps);
        return result;
    }

    /// <summary>Strips an [B, C, H, W] image into [B, NumPatches, patchDim] patches.</summary>
    private void Patchify(Tensor<float> image, Span<float> dst, int batch)
    {
        var img = image.Data;
        int patch = _patchSize;
        int gridW = _imageSize / patch;
        int patchDim = _patchDim;
        int channels = _channels;

        for (int b = 0; b < batch; b++)
        {
            for (int i = 0; i < gridW; i++)
            {
                for (int j = 0; j < gridW; j++)
                {
                    int dstRow = ((b * _numPatches) + i * gridW + j) * patchDim;
                    int k = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        for (int yy = 0; yy < patch; yy++)
                        {
                            for (int xx = 0; xx < patch; xx++)
                            {
                                int row = i * patch + yy;
                                int col = j * patch + xx;
                                int src = ((b * channels + c) * _imageSize + row) * _imageSize + col;
                                dst[dstRow + k++] = img[src];
                            }
                        }
                    }
                }
            }
        }
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter("vision.patch_proj.weight", _proj.Weight);
        yield return new Parameter("vision.pos_embed", _posEmbed);
        yield return new Parameter("vision.norm.weight", _normWeight);
    }

    internal static void ApplyRmsNorm(Tensor<float> x, ReadOnlySpan<float> weight, float eps)
    {
        var data = x.Data;
        int dim = x.Shape[^1];
        int rows = x.ElementCount / dim;
        for (int r = 0; r < rows; r++)
        {
            int off = r * dim;
            double sq = 0;
            for (int k = 0; k < dim; k++) sq += (double)data[off + k] * data[off + k];
            float inv = (float)(1.0 / Math.Sqrt(sq / dim + eps));
            for (int k = 0; k < dim; k++)
                data[off + k] = (data[off + k] * inv) * weight[k];
        }
    }

    internal static void FillUniformSmall(Span<float> span, Random rng)
    {
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _proj.Dispose();
        _posEmbed.Dispose();
        _normWeight.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(VisionEncoder));
}