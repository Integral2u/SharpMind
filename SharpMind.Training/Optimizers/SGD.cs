using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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

    public void LoadStateForWarmRestart(
        BinaryReader reader, int step, IReadOnlyList<ParameterLayout> checkpointParameters)
    {
        reader.ReadInt32(); // stored step; caller supplies the authoritative value
        _step = step;
        _lr = reader.ReadSingle();
        bool hasVelocity = reader.ReadBoolean();

        var targetIndices = new Dictionary<string, Queue<int>>(StringComparer.Ordinal);
        for (int i = 0; i < _parameters.Length; i++)
        {
            if (!targetIndices.TryGetValue(_parameters[i].Name, out var queue))
            {
                queue = new Queue<int>();
                targetIndices[_parameters[i].Name] = queue;
            }
            queue.Enqueue(i);
        }

        foreach (var source in checkpointParameters)
        {
            int sourceCount = checked((int)source.ElementCount);
            var sourceVelocity = new float[sourceCount];
            if (hasVelocity)
                ReadFloats(reader, sourceVelocity, $"SGD velocity for '{source.Name}'");

            if (!hasVelocity || _velocity is null) continue;
            if (!targetIndices.TryGetValue(source.Name, out var targets) || targets.Count == 0)
                continue;
            int targetIndex = targets.Dequeue();
            var targetVelocity = _velocity[targetIndex];
            int[] targetShape = _parameters[targetIndex].Data.Shape.Dims.ToArray();
            int[] sourceShape = source.Shape;

            if (SameShape(sourceShape, targetShape))
            {
                sourceVelocity.CopyTo(targetVelocity, 0);
                continue;
            }
            if (sourceShape.Length == 2 && targetShape.Length == 2
                && sourceShape[1] == targetShape[1]
                && targetShape[0] > sourceShape[0])
            {
                Array.Copy(sourceVelocity, 0, targetVelocity, 0, sourceShape[0] * sourceShape[1]);
                continue;
            }

            throw new InvalidDataException(
                $"Cannot warm-restart optimizer state for '{source.Name}': checkpoint shape " +
                $"[{string.Join(',', sourceShape)}] does not fit model shape " +
                $"[{string.Join(',', targetShape)}].");
        }
    }

    private static bool SameShape(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    private static void ReadFloats(BinaryReader reader, float[] destination, string what)
    {
        try
        {
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(destination.AsSpan()));
        }
        catch (EndOfStreamException)
        {
            throw new InvalidDataException($"Unexpected end of checkpoint while reading {what}.");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _parameters) p.Dispose();
    }
}