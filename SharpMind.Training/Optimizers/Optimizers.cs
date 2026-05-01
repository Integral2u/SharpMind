using SharpMind.Training.Autograd;

namespace SharpMind.Training.Optimizers;

/// <summary>
/// AdamW optimizer (Loshchilov &amp; Hutter, 2019).
///
/// The de-facto standard for LLM pre-training. Decouples weight decay from
/// the gradient adaptation — weight decay is applied directly to parameters
/// rather than being folded into the gradient update as in Adam+L2.
///
/// Update rule per parameter:
///   m = β₁ * m + (1 - β₁) * g                 (first moment)
///   v = β₂ * v + (1 - β₂) * g²                (second moment)
///   m̂ = m / (1 - β₁^t)                         (bias correction)
///   v̂ = v / (1 - β₂^t)                         (bias correction)
///   θ = θ - lr * (m̂ / (√v̂ + ε) + λ * θ)        (update + weight decay)
/// </summary>
public sealed class AdamW : IDisposable
{
    private readonly Parameter[]  _parameters;
    private readonly float[][]    _m;       // first moment vectors
    private readonly float[][]    _v;       // second moment vectors
    private readonly float        _beta1;
    private readonly float        _beta2;
    private readonly float        _epsilon;
    private readonly float        _weightDecay;
    private readonly HashSet<string> _noDecayNames;
    private float                 _lr;
    private int                   _step;
    private bool                  _disposed;

    /// <param name="parameters">All trainable parameters.</param>
    /// <param name="lr">Initial learning rate. Typically 1e-4 to 3e-4 for LLMs.</param>
    /// <param name="beta1">First moment decay. Default 0.9.</param>
    /// <param name="beta2">Second moment decay. Default 0.95 (LLaMA) or 0.999 (GPT).</param>
    /// <param name="epsilon">Numerical stability constant. Default 1e-8.</param>
    /// <param name="weightDecay">L2 weight decay coefficient. Default 0.1.</param>
    /// <param name="noDecayNames">
    /// Parameter name suffixes or substrings that should NOT have weight decay applied.
    /// Typically bias terms and norm weights. Defaults to "bias", "weight" for norms.
    /// </param>
    public AdamW(
        IEnumerable<Parameter> parameters,
        float                  lr           = 3e-4f,
        float                  beta1        = 0.9f,
        float                  beta2        = 0.95f,
        float                  epsilon      = 1e-8f,
        float                  weightDecay  = 0.1f,
        IEnumerable<string>?   noDecayNames = null)
    {
        _parameters   = [.. parameters];
        _lr           = lr;
        _beta1        = beta1;
        _beta2        = beta2;
        _epsilon      = epsilon;
        _weightDecay  = weightDecay;
        _noDecayNames = new HashSet<string>(
            noDecayNames ?? ["bias", "norm.weight", "layernorm.weight"],
            StringComparer.OrdinalIgnoreCase);

        _m = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
        _v = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
    }

    public float LearningRate
    {
        get => _lr;
        set => _lr = value;
    }

    public int Step => _step;

    /// <summary>
    /// Applies one AdamW update step to all parameters.
    /// Call after the backward pass; call <see cref="ZeroGrad"/> afterwards.
    /// </summary>
    public void Update()
    {
        _step++;
        float beta1T = MathF.Pow(_beta1, _step);
        float beta2T = MathF.Pow(_beta2, _step);
        float bc1    = 1f - beta1T;
        float bc2    = 1f - beta2T;

        for (int i = 0; i < _parameters.Length; i++)
        {
            var   param = _parameters[i];
            var   data  = param.Data.Data;
            var   grad  = param.Grad.Data;
            var   m     = _m[i].AsSpan();
            var   v     = _v[i].AsSpan();
            bool  decay = ShouldDecay(param.Name);

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

    /// <summary>Zeros all parameter gradients. Call after <see cref="Update"/>.</summary>
    public void ZeroGrad()
    {
        foreach (var p in _parameters) p.ZeroGrad();
    }

    private bool ShouldDecay(string name)
    {
        foreach (string noDecay in _noDecayNames)
            if (name.Contains(noDecay, StringComparison.OrdinalIgnoreCase))
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

/// <summary>
/// Stochastic gradient descent with optional momentum and weight decay.
/// Simpler than AdamW; useful for fine-tuning or as a baseline.
/// θ = θ - lr * (g + λ * θ)  [with optional momentum]
/// </summary>
public sealed class SGD : IDisposable
{
    private readonly Parameter[] _parameters;
    private readonly float[][]?  _velocity;
    private readonly float       _momentum;
    private readonly float       _weightDecay;
    private float                _lr;
    private bool                 _disposed;

    public SGD(
        IEnumerable<Parameter> parameters,
        float                  lr          = 1e-3f,
        float                  momentum    = 0f,
        float                  weightDecay = 0f)
    {
        _parameters  = [.. parameters];
        _lr          = lr;
        _momentum    = momentum;
        _weightDecay = weightDecay;

        if (momentum > 0f)
            _velocity = [.. _parameters.Select(p => new float[p.Data.ElementCount])];
    }

    public float LearningRate { get => _lr; set => _lr = value; }

    public void Update()
    {
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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var p in _parameters) p.Dispose();
    }
}
