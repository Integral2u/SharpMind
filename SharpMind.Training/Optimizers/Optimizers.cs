using System.IO;
using System.Runtime.CompilerServices;
using SharpMind.Core.Training;

namespace SharpMind.Training.Optimizers;

public sealed class AdamW : IOptimizer
{
    private readonly Parameter[] _parameters;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly HashSet<string> _noDecayNames;
    private float _lr;
    private int _step;
    private bool _disposed;

    public AdamW(
        IEnumerable<Parameter> parameters,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float epsilon = 1e-8f,
        float weightDecay = 0.1f,
        IEnumerable<string>? noDecayNames = null)
    {
        _parameters = [.. parameters];
        _lr = lr;
        _beta1 = beta1;
        _beta2 = beta2;
        _epsilon = epsilon;
        _weightDecay = weightDecay;
        _noDecayNames = new HashSet<string>(
            noDecayNames ?? ["bias", "norm.weight", "layernorm.weight"],
            StringComparer.OrdinalIgnoreCase);

        _m = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
        _v = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
    }

    public float LearningRate { get => _lr; set => _lr = value; }
    public int Step => _step;

    public void Update()
    {
        _step++;
        float bc1 = 1f - MathF.Pow(_beta1, _step);
        float bc2 = 1f - MathF.Pow(_beta2, _step);

        for (int i = 0; i < _parameters.Length; i++)
        {
            var data = _parameters[i].Data.Data;
            var grad = _parameters[i].Grad.Data;
            var m = _m[i];
            var v = _v[i];
            bool decay = ShouldDecay(_parameters[i].Name);

            for (int j = 0; j < data.Length; j++)
            {
                float g = grad[j];
                m[j] = _beta1 * m[j] + (1f - _beta1) * g;
                v[j] = _beta2 * v[j] + (1f - _beta2) * g * g;

                float mHat = m[j] / bc1;
                float vHat = v[j] / bc2;

                float update = _lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
                if (decay) update += _lr * _weightDecay * data[j];
                data[j] -= update;
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
        for (int i = 0; i < _m.Length; i++)
        {
            foreach (float f in _m[i]) writer.Write(f);
            foreach (float f in _v[i]) writer.Write(f);
        }
    }

    public void LoadState(BinaryReader reader, int step)
    {
        _step = step;
        _lr = reader.ReadSingle();
        for (int i = 0; i < _m.Length; i++)
        {
            for (int j = 0; j < _m[i].Length; j++) _m[i][j] = reader.ReadSingle();
            for (int j = 0; j < _v[i].Length; j++) _v[i][j] = reader.ReadSingle();
        }
    }

    private bool ShouldDecay(string name)
    {
        foreach (string nd in _noDecayNames)
            if (name.Contains(nd, StringComparison.OrdinalIgnoreCase))
                return false;
        return true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _parameters) p.Dispose();
    }
}

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