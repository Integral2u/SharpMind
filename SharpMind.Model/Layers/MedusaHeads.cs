using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public sealed class MedusaHeads : IDisposable
{
    private readonly int _numHeads;
    private readonly int _hiddenDim;
    private readonly int _vocabSize;
    private readonly Tensor<float>[] _headWeights;
    private readonly Tensor<float>[] _headBiases;
    private readonly Tensor<float> _lmHeadWeight;
    private readonly byte[]? _rawEmbedding;
    private readonly QuantDType? _rawDtype;
    private readonly QuantizationOps? _qOps;
    private float[]? _logitBuffer;
    private float[]? _predictLogitBuffer;
    private bool _disposed;

    public MedusaHeads(int numHeads, int hiddenDim, int vocabSize, Tensor<float> lmHeadWeight,
        byte[]? rawEmbedding = null, QuantDType? rawDtype = null, QuantizationOps? qOps = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numHeads);
        _numHeads = numHeads;
        _hiddenDim = hiddenDim;
        _vocabSize = vocabSize;
        _lmHeadWeight = lmHeadWeight;
        _rawEmbedding = rawEmbedding;
        _rawDtype = rawDtype;
        _qOps = qOps;

        _headWeights = new Tensor<float>[numHeads];
        _headBiases = new Tensor<float>[numHeads];

        for (int i = 0; i < numHeads; i++)
        {
            _headWeights[i] = new Tensor<float>(hiddenDim, hiddenDim);
            var rng = new Random(i * 42 + 1);
            float scale = MathF.Sqrt(2.0f / hiddenDim);
            var wData = _headWeights[i].Data;
            for (int j = 0; j < wData.Length; j++)
                wData[j] = (float)(rng.NextDouble() * 2 - 1) * scale;

            _headBiases[i] = new Tensor<float>(hiddenDim);
        }

        IsTrained = false;
    }

    public int NumHeads => _numHeads;
    public bool IsTrained { get; private set; }

    /// <summary>Access head weights for training.</summary>
    public Tensor<float> GetHeadWeight(int head) => _headWeights[head];
    public Tensor<float> GetHeadBias(int head) => _headBiases[head];

    /// <summary>
    /// Given a normed hidden state vector [1, HiddenDim], predicts draft tokens
    /// for offsets 0..K-1 (offset 0 = immediate next token, same as LM head).
    /// Returns K draft token IDs in outputTokens.
    /// </summary>
    public void Predict(Span<float> hiddenState, Span<int> outputTokens)
    {
        int k = Math.Min(outputTokens.Length, _numHeads);
        if (k == 0) return;

        if (_rawEmbedding != null && _rawDtype == QuantDType.Q8_0 && _qOps != null)
        {
            PredictQuantized(hiddenState, outputTokens, k);
        }
        else
        {
            PredictFloat(hiddenState, outputTokens, k);
        }
    }

    private unsafe void PredictQuantized(Span<float> hiddenState, Span<int> outputTokens, int k)
    {
        float* headsInput = stackalloc float[k * _hiddenDim];

        for (int h = 0; h < k; h++)
        {
            var w = _headWeights[h];
            var b = _headBiases[h];
            float* pOut = headsInput + (long)h * _hiddenDim;

            for (int i = 0; i < _hiddenDim; i++)
            {
                float sum = b.Data[i];
                for (int j = 0; j < _hiddenDim; j++)
                    sum += hiddenState[j] * w.Data[j * _hiddenDim + i];
                pOut[i] = sum;
            }

            for (int i = 0; i < _hiddenDim; i++)
                pOut[i] = pOut[i] / (1.0f + MathF.Exp(-pOut[i]));
        }

        int logitLen = k * _vocabSize;
        if (_predictLogitBuffer == null || _predictLogitBuffer.Length < logitLen)
            _predictLogitBuffer = GC.AllocateUninitializedArray<float>(logitLen);

        fixed (byte* pRaw = _rawEmbedding!)
        fixed (float* pLogits = _predictLogitBuffer)
        {
            _qOps!.QuantizedMatMulQ8_0(headsInput, pRaw, pLogits, k, _hiddenDim, _vocabSize);

            for (int h = 0; h < k; h++)
            {
                float* row = pLogits + (long)h * _vocabSize;
                int bestIdx = 0;
                float bestVal = row[0];
                for (int v = 1; v < _vocabSize; v++)
                {
                    if (row[v] > bestVal) { bestVal = row[v]; bestIdx = v; }
                }
                outputTokens[h] = bestIdx;
            }
        }
    }

    private void PredictFloat(Span<float> hiddenState, Span<int> outputTokens, int k)
    {
        using var headsTensor = new Tensor<float>(k, _hiddenDim);
        for (int h = 0; h < k; h++)
        {
            var w = _headWeights[h];
            var b = _headBiases[h];
            var row = headsTensor.RowSpan(h);

            for (int i = 0; i < _hiddenDim; i++)
            {
                float sum = b.Data[i];
                for (int j = 0; j < _hiddenDim; j++)
                    sum += hiddenState[j] * w.Data[j * _hiddenDim + i];
                row[i] = sum;
            }

            for (int i = 0; i < _hiddenDim; i++)
                row[i] = row[i] / (1.0f + MathF.Exp(-row[i]));
        }

        using var logits = new Tensor<float>(k, _vocabSize);
        unsafe
        {
            var fn = _qOps!.QuantizedMatMulOpFor(QuantDType.F32);
            fn(headsTensor.DataPtr, (byte*)_lmHeadWeight.DataPtr, logits.DataPtr, k, _hiddenDim, _vocabSize);
        }

        for (int h = 0; h < k; h++)
        {
            var row = logits.Data.Slice(h * _vocabSize, _vocabSize);
            int bestIdx = 0;
            float bestVal = row[0];
            for (int v = 1; v < _vocabSize; v++)
            {
                if (row[v] > bestVal) { bestVal = row[v]; bestIdx = v; }
            }
            outputTokens[h] = bestIdx;
        }
    }

    private unsafe void ComputeLogitsQ8(float[] hPrime)
    {
        fixed (float* pIn = hPrime)
        fixed (byte* pRaw = _rawEmbedding)
        fixed (float* pOut = _logitBuffer)
        {
            _qOps!.QuantizedMatMulQ8_0(pIn, pRaw, pOut, 1, _hiddenDim, _vocabSize);
        }
    }

    /// <summary>
    /// Self-calibration: trains the heads to predict the model's own greedy output.
    /// Uses the model directly: feeds draft tokens and checks predictions.
    /// Parameters:
    ///   hiddenStates - [numSamples, HiddenDim] collected from model forward passes
    ///   targetTokens - [numSamples, K] target token IDs for each head offset
    ///   learningRate - SGD step size
    ///   steps - number of training epochs over the data
    /// </summary>
    public void Calibrate(Tensor<float> hiddenStates, int[][] targetTokens,
        float learningRate = 0.01f, int steps = 50)
    {
        int numSamples = hiddenStates.Shape.Rows;
        int targetLen = targetTokens[0].Length;
        int k = Math.Min(_numHeads, targetLen);

        var lossAccum = new float[k];
        var preAct = new float[_hiddenDim];
        var hPrime = new float[_hiddenDim];
        var dHPrime = new float[_hiddenDim];
        if (_logitBuffer == null || _logitBuffer.Length != _vocabSize)
            _logitBuffer = new float[_vocabSize];

        bool useQ8 = _rawEmbedding != null && _rawDtype == QuantDType.Q8_0 && _qOps != null;
        var lmData = _lmHeadWeight.Data;

        for (int step = 0; step < steps; step++)
        {
            Array.Clear(lossAccum);

            for (int s = 0; s < numSamples; s++)
            {
                var hRow = hiddenStates.RowSpan(s);

                for (int head = 0; head < k; head++)
                {
                    int targetId = targetTokens[s][head];
                    var w = _headWeights[head];
                    var b = _headBiases[head];

                    // Forward: h' = SiLU(h @ W + b), save pre-activation for backward
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        float sum = b.Data[i];
                        for (int j = 0; j < _hiddenDim; j++)
                            sum += hRow[j] * w.Data[j * _hiddenDim + i];
                        preAct[i] = sum;
                        hPrime[i] = sum / (1.0f + MathF.Exp(-sum));
                    }

                    // Compute all logits in ONE pass
                    if (useQ8)
                    {
                        ComputeLogitsQ8(hPrime);
                    }
                    else
                    {
                        int v = 0;
                        for (; v < _vocabSize; v++)
                        {
                            double sum = 0;
                            var row = lmData.Slice(v * _hiddenDim, _hiddenDim);
                            for (int i = 0; i < _hiddenDim; i++)
                                sum += hPrime[i] * row[i];
                            _logitBuffer[v] = (float)sum;
                        }
                    }

                    // Find maxLogit and targetLogit from buffer
                    double maxLogit = _logitBuffer[0];
                    double targetLogit = 0;
                    for (int v = 0; v < _vocabSize; v++)
                    {
                        double l = _logitBuffer[v];
                        if (l > maxLogit) maxLogit = l;
                        if (v == targetId) targetLogit = l;
                    }

                    // Softmax denominator
                    double sumExp = 0, targetExp = 0;
                    for (int v = 0; v < _vocabSize; v++)
                    {
                        double e = Math.Exp(_logitBuffer[v] - maxLogit);
                        sumExp += e;
                        if (v == targetId) targetExp = e;
                    }

                    double loss = -Math.Log(targetExp / sumExp);
                    lossAccum[head] += (float)loss;

                    // Gradient w.r.t. hPrime (from stored logits — no LM head recomputation)
                    Array.Clear(dHPrime);
                    double invSum = 1.0 / sumExp;
                    for (int v = 0; v < _vocabSize; v++)
                    {
                        double softmax = Math.Exp(_logitBuffer[v] - maxLogit) * invSum;
                        double d = softmax - (v == targetId ? 1.0 : 0.0);
                        var row = lmData.Slice(v * _hiddenDim, _hiddenDim);
                        for (int i = 0; i < _hiddenDim; i++)
                            dHPrime[i] += (float)(d * row[i]);
                    }

                    // SiLU backward + weight update (using saved preAct, no recomputation)
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        float sig = 1.0f / (1.0f + MathF.Exp(-preAct[i]));
                        float siluGrad = sig + preAct[i] * sig * (1.0f - sig);
                        float lrDo = learningRate * dHPrime[i] * siluGrad;

                        for (int j = 0; j < _hiddenDim; j++)
                            w.Data[j * _hiddenDim + i] -= lrDo * hRow[j];
                        b.Data[i] -= lrDo;
                    }
                }
            }

            if (step % 10 == 9)
            {
                float avgLoss = lossAccum.Sum() / (k * numSamples);
                System.Diagnostics.Debug.WriteLine($"[Medusa] Step {step + 1}/{steps}, avg loss: {avgLoss:F4}");
            }
        }

        IsTrained = true;
    }

    /// <summary>
    /// Self-calibration: collects training data from the model's own greedy
    /// outputs, then trains the heads via <see cref="Calibrate"/>.
    ///
    /// Runs <paramref name="numSamples"/> single-token forward steps on the
    /// model, saves the normed hidden state and the greedy next token at each
    /// step.  The K subsequent tokens become the K-head training targets for
    /// that step (head i must predict the token at offset i+1).
    ///
    /// Temporary workspace and KV caches are created internally and disposed
    /// when the method returns.  The model's own caches are NOT modified.
    /// </summary>
    /// <param name="model">The transformer model to calibrate against.</param>
    /// <param name="numSamples">Number of training samples to collect (default 50).</param>
    /// <param name="trainingSteps">SGD epochs over the collected data (default 30).</param>
    /// <param name="learningRate">SGD step size (default 0.01).</param>
    public void SelfCalibrate(Transformer model, int numSamples = 50,
        int trainingSteps = 30, float learningRate = 0.01f)
    {
        if (_numHeads == 0) return;
        int K = _numHeads;
        int totalSteps = numSamples + K; // K extra for valid K-head targets

        // Create workspace sized for a single decode step × totalSteps horizon.
        using var workspace = new Workspace(
            Workspace.CalculateRequiredSize(
                model.Config.HiddenDim, model.Config.FfnDim,
                model.Config.VocabSize, model.Config.NumLayers,
                totalSteps));

        // Create independent KV caches so the model's own cache is untouched.
        var caches = new IKVCache[model.Config.NumLayers];
        try
        {
            for (int i = 0; i < caches.Length; i++)
                caches[i] = new KVCache(1, model.Config.NumKvHeads,
                    totalSteps, model.Config.HeadDim);

            // Storage: one hidden vector per sample, K target token IDs per sample.
            var hiddenStates = new Tensor<float>(numSamples, _hiddenDim);
            var targetTokens = new int[numSamples][];
            for (int i = 0; i < numSamples; i++)
                targetTokens[i] = new int[K];

            // Greedy autoregressive generation.
            // Each step processes one token, caches the key/value, and saves
            // the normed hidden state and the greedy next-token ID.
            var generated = new int[totalSteps];
            for (int step = 0; step < totalSteps; step++)
            {
                workspace.Reset();
                using var input = workspace.Rent<int>([1, 1]);
                input.Data[0] = step == 0 ? 0 : generated[step - 1];

                using var logits = model.ForwardLastLogits(
                    input, caches, step, workspace);

                // ArgMax for greedy token
                float bestVal = logits.Data[0];
                int bestIdx = 0;
                for (int v = 1; v < _vocabSize; v++)
                {
                    if (logits.Data[v] <= bestVal) continue;
                    bestVal = logits.Data[v];
                    bestIdx = v;
                }
                generated[step] = bestIdx;

                // Save the normed hidden state for this position.
                // After single-token ForwardLastLogits, _cachedHidden is
                // [1, 1, H] and already final-normed (ForwardInPlace path).
                if (step < numSamples)
                {
                    var ch = model.LastCachedHidden;
                    ch?.Data[.._hiddenDim].CopyTo(hiddenStates.RowSpan(step));
                }
            }

            // Fill target tokens: for position P, head i must predict the
            // token at offset i+1 ahead, which is generated[P + i].
            for (int P = 0; P < numSamples; P++)
                for (int i = 0; i < K; i++)
                    targetTokens[P][i] = generated[P + i];

            // Train the heads on the collected data.
            Calibrate(hiddenStates, targetTokens, learningRate, trainingSteps);
            hiddenStates.Dispose();
        }
        finally
        {
            foreach (var c in caches)
                c?.Dispose();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var w in _headWeights) w.Dispose();
        foreach (var b in _headBiases) b.Dispose();
    }
}
