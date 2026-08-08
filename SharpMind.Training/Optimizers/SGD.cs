using System.Runtime.CompilerServices;
using SharpMind.Core.Training;

namespace SharpMind.Training.Optimizers;

public sealed class SGD : IOptimizer
{
    private readonly Parameter[] _parameters;
    private readonly float[][]? _velocity;
    private readonly float _momentum;
    private readonly float _weightDecay;
    private float _lr;
    private int _step;
    private bool _disposed;

    public SGD(
        IEnumerable<Parameter> parameters,
        float lr = 1e-3f,
        float momentum = 0f,
        float weightDecay = 0f)
    {
        _parameters = [.. parameters];
        _lr = lr;
        _momentum = momentum;
        _weightDecay = weightDecay;

        if (momentum > 0f)
            _velocity = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
    }

    public float LearningRate { get => _lr; set => _lr = value; }
    public int Step => _step;
    public IReadOnlyList<Parameter> Parameters => _parameters;

    /// <summary>Applies one SGD update step.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Update()
    {
        _step++;
        for (int i = 0; i < _parameters.Length; i++)
        {
            var data = _parameters[i].Data.Data;
            var grad = _parameters[i].Grad.Data;

            for (int j = 0; j < data.Length; j++)
            {
                float g = grad[j] + _weightDecay * data[j];

                if (_velocity is not null)
                {
                    _velocity[i][j] = _momentum * _velocity[i][j] + g;
                    g = _velocity[i][j];
                }

                data[j] -= _lr * g;
            }
        }
    }

    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.ZeroGrad();
    }

    public void SaveState(BinaryWriter writer)
    {
        writer.Write(_step);
        writer.Write(_lr);
        if (_velocity is not null)
        {
            writer.Write(true);
            for (int i = 0; i < _velocity.Length; i++)
                foreach (float f in _velocity[i]) writer.Write(f);
        }
        else
        {
            writer.Write(false);
        }
    }

    public void LoadState(BinaryReader reader, int step)
    {
        reader.ReadInt32(); // skip stored _step; caller supplies the authoritative value
        _step = step;
        _lr = reader.ReadSingle();
        bool hasVelocity = reader.ReadBoolean();
        if (hasVelocity && _velocity is not null)
        {
            for (int i = 0; i < _velocity.Length; i++)
                for (int j = 0; j < _velocity[i].Length; j++)
                    _velocity[i][j] = reader.ReadSingle();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _parameters) p.Dispose();
    }
}