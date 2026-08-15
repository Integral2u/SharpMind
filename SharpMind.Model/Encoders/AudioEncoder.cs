using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Encoders;

/// <summary>
/// Audio encoder: turns a mel-spectrogram into a sequence of HiddenDim frame
/// embeddings that a decoder transformer can attend to.
///
/// Pipeline:
///   mel [B, Frames, MelBins]
///     → linear projection → [B*Frames, HiddenDim] → [B, Frames, HiddenDim]
///     → learned positional embedding + RMS norm
///     → output [B, Frames, HiddenDim]
///
/// The encoder owns its frame projection, position embedding and norm
/// parameters, exposed through <see cref="Parameters"/> for a future optimizer.
/// </summary>
public sealed class AudioEncoder : IDisposable
{
    private readonly int _melBins;
    private readonly int _maxFrames;
    private readonly int _hiddenDim;
    private readonly LinearLayer _proj;
    private readonly Tensor<float> _posEmbed;   // [MaxFrames, HiddenDim]
    private readonly Tensor<float> _normWeight; // [HiddenDim]
    private readonly float _eps;
    private bool _disposed;

    public AudioEncoder(ModelConfig config, Random? rng = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!config.HasAudio)
            throw new ArgumentException("Audio encoder requires AudioMelBins.", nameof(config));
        config.Validate();

        rng ??= Random.Shared;
        _melBins = config.AudioMelBins!.Value;
        _maxFrames = config.AudioMaxFrames;
        _hiddenDim = config.HiddenDim;
        _eps = config.NormEps;

        _proj = LinearLayerFactory.Create(
            "audio.frame_proj", _melBins, _hiddenDim, bias: false,
            weight: null, biasTensor: null, QuantDType.F32);

        _posEmbed = new Tensor<float>(_maxFrames, _hiddenDim);
        VisionEncoder.FillUniformSmall(_posEmbed.Data, rng);

        _normWeight = Tensor<float>.Ones(_hiddenDim);
    }

    /// <summary>Mel-bin count of a single input frame.</summary>
    public int InFeatures => _melBins;
    /// <summary>Maximum number of audio frames the encoder supports.</summary>
    public int MaxFrames => _maxFrames;
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
    /// Embeds a mel-spectrogram into frame tokens.
    /// Input:  [Batch, Frames, MelBins]
    /// Output: [Batch, Frames, HiddenDim]
    /// </summary>
    public Tensor<float> Forward(Tensor<float> mel)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(mel);
        if (mel.Rank != 3)
            throw new ArgumentException($"Audio encoder expects 3D mel [B, Frames, MelBins], got rank {mel.Rank}.");
        if (mel.Shape[2] != _melBins)
            throw new ArgumentException($"Audio encoder expects {_melBins} mel bins, got {mel.Shape[2]}.");
        if (mel.Shape[1] > _maxFrames)
            throw new ArgumentException(
                $"Audio encoder supports at most {_maxFrames} frames, got {mel.Shape[1]}.");

        int batch = mel.Shape[0];
        int frames = mel.Shape[1];

        // Frame projection on the flattened [B*Frames, MelBins] input.
        using var flat = mel.Reshape(batch * frames, _melBins);
        using var projected2d = _proj.Forward(flat);
        using var projected = projected2d.Reshape(batch, frames, _hiddenDim);

        // Add positional embeddings, then RMS-normalise.
        var result = new Tensor<float>(batch, frames, _hiddenDim);
        var resultData = result.Data;
        var posData = _posEmbed.Data;
        int hidden = _hiddenDim;
        projected.Data.CopyTo(resultData);
        for (int b = 0; b < batch; b++)
        {
            int rowBase = b * frames * hidden;
            for (int f = 0; f < frames; f++)
            {
                int rowOff = rowBase + f * hidden;
                for (int h = 0; h < hidden; h++)
                    resultData[rowOff + h] += posData[f * hidden + h];
            }
        }

        VisionEncoder.ApplyRmsNorm(result, _normWeight.Data, _eps);
        return result;
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter("audio.frame_proj.weight", _proj.Weight);
        yield return new Parameter("audio.pos_embed", _posEmbed);
        yield return new Parameter("audio.norm.weight", _normWeight);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _proj.Dispose();
        _posEmbed.Dispose();
        _normWeight.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AudioEncoder));
}