using System.Runtime.InteropServices;
using SharpMind.Core.Training;

namespace SharpMind.Training.Optimizers;

public sealed class AdamW : IOptimizer
{
    private readonly Parameter[] _parameters;
    private readonly float[][] _m;
    private readonly float[][] _v;
    private readonly bool[] _decay;
    private readonly float _beta1;
    private readonly float _beta2;
    private readonly float _epsilon;
    private readonly float _weightDecay;
    private readonly HashSet<string> _noDecayNames;
    private readonly TrainingOps? _ops;
    private float _lr;
    private int _step;
    private bool _disposed;

    public AdamW(
        IEnumerable<Parameter> parameters,
        TrainingOps? ops = null,
        float lr = 3e-4f,
        float beta1 = 0.9f,
        float beta2 = 0.95f,
        float epsilon = 1e-8f,
        float weightDecay = 0.1f,
        IEnumerable<string>? noDecayNames = null)
    {
        _parameters = [.. parameters];
        _ops = ops;
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
        _decay = [.. _parameters.Select(p => ShouldDecay(p.Name))];
    }

    public float LearningRate { get => _lr; set => _lr = value; }
    public int Step => _step;
    public IReadOnlyList<Parameter> Parameters => _parameters;

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
            float decay = _decay[i] ? _weightDecay : 0f;

            if (_ops is not null)
            {
                _ops.AdamWUpdate(data, grad, m, v,
                    _beta1, _beta2, bc1, bc2,
                    _lr, _epsilon, decay);
            }
            else
            {
                for (int j = 0; j < data.Length; j++)
                {
                    float g = grad[j];
                    m[j] = _beta1 * m[j] + (1f - _beta1) * g;
                    v[j] = _beta2 * v[j] + (1f - _beta2) * g * g;

                    float mHat = m[j] / bc1;
                    float vHat = v[j] / bc2;

                    float update = _lr * mHat / (MathF.Sqrt(vHat) + _epsilon);
                    if (decay > 0f) update += _lr * decay * data[j];
                    data[j] -= update;
                }
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
            writer.Write(MemoryMarshal.AsBytes(_m[i].AsSpan()));
            writer.Write(MemoryMarshal.AsBytes(_v[i].AsSpan()));
        }
    }

    public void LoadState(BinaryReader reader, int step)
    {
        reader.ReadInt32(); // skip stored _step; caller supplies the authoritative value
        _step = step;
        _lr = reader.ReadSingle();
        for (int i = 0; i < _m.Length; i++)
        {
            reader.Read(MemoryMarshal.AsBytes(_m[i].AsSpan()));
            reader.Read(MemoryMarshal.AsBytes(_v[i].AsSpan()));
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
