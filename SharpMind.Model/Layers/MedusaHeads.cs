using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

/// <summary>Outcome of a Medusa self-calibration pass.</summary>
/// <param name="Samples">Number of greedy samples collected from the model.</param>
/// <param name="Steps">Number of SGD epochs run over those samples.</param>
/// <param name="FinalLoss">Average cross-entropy over the last epoch, or 0 when nothing was trained.</param>
public readonly record struct MedusaCalibrationResult(int Samples, int Steps, float FinalLoss);

/// <summary>Which half of a Medusa calibration pass is running.</summary>
public enum MedusaCalibrationPhase
{
    /// <summary>Greedy continuation is being generated to build the training set.</summary>
    Collecting,

    /// <summary>Gradients are being accumulated and the heads updated.</summary>
    Training,
}

/// <summary>Progress of a Medusa self-calibration pass.</summary>
/// <param name="Phase">Current phase.</param>
/// <param name="Completed">Steps finished in this phase.</param>
/// <param name="Total">Steps in this phase.</param>
/// <param name="Loss">Average loss of the last finished epoch, or null while collecting.</param>
public readonly record struct MedusaCalibrationProgress(
    MedusaCalibrationPhase Phase,
    int Completed,
    int Total,
    float? Loss);

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
    public int HiddenDim => _hiddenDim;
    public int VocabSize => _vocabSize;
    public bool IsTrained { get; private set; }

    /// <summary>Result of the last <see cref="SelfCalibrate"/> call, or null when never calibrated.</summary>
    public MedusaCalibrationResult? LastCalibration { get; private set; }

    /// <summary>Access head weights for training.</summary>
    public Tensor<float> GetHeadWeight(int head) => _headWeights[head];
    public Tensor<float> GetHeadBias(int head) => _headBiases[head];

    /// <summary>
    /// Replaces the random heads with a set loaded by <see cref="MedusaHeadFile"/>.
    /// Shapes and an optional fingerprint are checked, so heads trained against a
    /// different model fail loudly instead of quietly drafting nonsense.
    /// </summary>
    /// <exception cref="InvalidDataException">The snapshot does not match this head set.</exception>
    public void LoadWeights(MedusaHeadSnapshot snapshot, string? expectedFingerprint = null)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        if (snapshot.NumHeads != _numHeads
            || snapshot.HiddenDim != _hiddenDim
            || snapshot.VocabSize != _vocabSize)
        {
            throw new InvalidDataException(
                $"Medusa head file is {snapshot.NumHeads} heads / {snapshot.HiddenDim} hidden / " +
                $"{snapshot.VocabSize} vocab, but the model needs {_numHeads} / {_hiddenDim} / {_vocabSize}.");
        }

        if (expectedFingerprint is not null
            && !string.Equals(expectedFingerprint, snapshot.Fingerprint, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Medusa head file fingerprint '{snapshot.Fingerprint}' does not match '{expectedFingerprint}'.");
        }

        if (snapshot.Weights.Length != _numHeads || snapshot.Biases.Length != _numHeads)
            throw new InvalidDataException("Medusa head file has the wrong number of weight arrays.");

        // Validate every tensor before copying any, so a bad file cannot leave
        // the head set half old and half new.
        for (int h = 0; h < _numHeads; h++)
        {
            if (snapshot.Weights[h].Length != _hiddenDim * _hiddenDim
                || snapshot.Biases[h].Length != _hiddenDim)
            {
                throw new InvalidDataException($"Medusa head {h} has the wrong tensor length.");
            }
        }

        for (int h = 0; h < _numHeads; h++)
        {
            snapshot.Weights[h].CopyTo(_headWeights[h].Data);
            snapshot.Biases[h].CopyTo(_headBiases[h].Data);
        }

        IsTrained = snapshot.IsTrained;
        LastCalibration = snapshot.Calibration;
    }

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
    /// <returns>Average cross-entropy over the final epoch.</returns>
    public float Calibrate(Tensor<float> hiddenStates, int[][] targetTokens,
        float learningRate = 0.01f, int steps = 50,
        CancellationToken cancellationToken = default,
        IProgress<MedusaCalibrationProgress>? progress = null)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int numSamples = hiddenStates.Shape.Rows;
        if (numSamples == 0 || targetTokens.Length == 0 || steps <= 0)
        {
            IsTrained = true;
            return 0f;
        }

        int targetLen = targetTokens[0].Length;
        int k = Math.Min(_numHeads, targetLen);
        if (k == 0)
        {
            IsTrained = true;
            return 0f;
        }

        var lossAccum = new float[k];
        var preAct = new float[_hiddenDim];
        var hPrime = new float[_hiddenDim];
        var dHPrime = new float[_hiddenDim];
        var gradW = new float[k][];
        var gradB = new float[k][];
        for (int h = 0; h < k; h++)
        {
            gradW[h] = new float[_hiddenDim * _hiddenDim];
            gradB[h] = new float[_hiddenDim];
        }

        if (_logitBuffer == null || _logitBuffer.Length != _vocabSize)
            _logitBuffer = new float[_vocabSize];

        bool useQ8 = _rawEmbedding != null && _rawDtype == QuantDType.Q8_0 && _qOps != null;
        var lmData = _lmHeadWeight.Data;
        float finalLoss = 0f;

        // The cross-entropy gradient through the LM head is large for a random
        // head (it scales with the vocabulary), so an unclipped per-sample SGD
        // step diverges within a few epochs. Accumulate one epoch, clip the
        // gradient norm, and apply one averaged update per epoch.
        const float MaxGradientNorm = 1f;

        for (int step = 0; step < steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Array.Clear(lossAccum);
            for (int h = 0; h < k; h++)
            {
                Array.Clear(gradW[h]);
                Array.Clear(gradB[h]);
            }

            for (int s = 0; s < numSamples; s++)
            {
                cancellationToken.ThrowIfCancellationRequested();
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
                        for (int v = 0; v < _vocabSize; v++)
                        {
                            double sum = 0;
                            var row = lmData.Slice(v * _hiddenDim, _hiddenDim);
                            for (int i = 0; i < _hiddenDim; i++)
                                sum += hPrime[i] * row[i];
                            _logitBuffer[v] = (float)sum;
                        }
                    }

                    // Find maxLogit from buffer for softmax stability
                    double maxLogit = _logitBuffer[0];
                    for (int v = 1; v < _vocabSize; v++)
                        if (_logitBuffer[v] > maxLogit) maxLogit = _logitBuffer[v];

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

                    // SiLU backward, accumulate the outer product for this head.
                    var gw = gradW[head];
                    var gb = gradB[head];
                    for (int i = 0; i < _hiddenDim; i++)
                    {
                        float sig = 1.0f / (1.0f + MathF.Exp(-preAct[i]));
                        float siluGrad = sig + preAct[i] * sig * (1.0f - sig);
                        float g = dHPrime[i] * siluGrad;

                        gb[i] += g;
                        for (int j = 0; j < _hiddenDim; j++)
                            gw[j * _hiddenDim + i] += g * hRow[j];
                    }
                }
            }

            for (int head = 0; head < k; head++)
            {
                var w = _headWeights[head];
                var b = _headBiases[head];
                var gw = gradW[head];
                var gb = gradB[head];

                double normSq = 0;
                for (int idx = 0; idx < gw.Length; idx++) normSq += gw[idx] * (double)gw[idx];
                for (int i = 0; i < gb.Length; i++) normSq += gb[i] * (double)gb[i];

                float clip = normSq > MaxGradientNorm * MaxGradientNorm
                    ? MaxGradientNorm / MathF.Sqrt((float)normSq)
                    : 1f;
                float scale = learningRate * clip;

                for (int idx = 0; idx < gw.Length; idx++)
                    w.Data[idx] -= scale * gw[idx];
                for (int i = 0; i < gb.Length; i++)
                    b.Data[i] -= scale * gb[i];
            }

            finalLoss = lossAccum.Sum() / (k * numSamples);
            System.Diagnostics.Debug.WriteLine($"[Medusa] Step {step + 1}/{steps}, avg loss: {finalLoss:F4}");
            progress?.Report(new MedusaCalibrationProgress(
                MedusaCalibrationPhase.Training, step + 1, steps, finalLoss));
        }

        IsTrained = true;
        return finalLoss;
    }

    /// <summary>
    /// Self-calibration: collects training data from the model's own greedy
    /// outputs, then trains the heads via <see cref="Calibrate"/>.
    ///
    /// Starts from <paramref name="seedTokens"/> (or token 0), runs a greedy
    /// continuation, then replays the whole sequence through the blocks and
    /// saves the normed hidden state at each sample position. The LM head
    /// supplies offset 1 (token0), so head i is trained to predict offset i+2.
    ///
    /// Temporary workspace and KV caches are created internally and disposed
    /// when the method returns.  The model's own caches are NOT modified.
    /// </summary>
    /// <param name="model">The transformer model to calibrate against.</param>
    /// <param name="numSamples">Number of training samples to collect (default 50).</param>
    /// <param name="trainingSteps">SGD epochs over the collected data (default 30).</param>
    /// <param name="learningRate">SGD step size (default 0.01).</param>
    /// <param name="seedTokens">
    /// Prompt/context tokens to start from. Null starts at token 0, which is fine
    /// for tests but a real prompt makes the heads match the text they will draft.
    /// </param>
    /// <param name="cancellationToken">Stops collection or training between steps.</param>
    /// <param name="progress">
    /// Optional sink for collection and per-epoch training progress. Reported from
    /// the calling thread; do not block in the handler.
    /// </param>
    /// <returns>The sample count, epoch count, and final training loss.</returns>
    public MedusaCalibrationResult SelfCalibrate(Transformer model, int numSamples = 50,
        int trainingSteps = 30, float learningRate = 0.01f, IReadOnlyList<int>? seedTokens = null,
        CancellationToken cancellationToken = default,
        IProgress<MedusaCalibrationProgress>? progress = null)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (_numHeads == 0 || numSamples <= 0 || trainingSteps <= 0)
        {
            var skipped = new MedusaCalibrationResult(
                Math.Max(0, numSamples), Math.Max(0, trainingSteps), 0f);
            LastCalibration = skipped;
            return skipped;
        }

        int K = _numHeads;
        int totalSteps = numSamples + K; // K extra for valid K-head targets
        int seedLen = seedTokens is { Count: > 0 } ? seedTokens.Count : 1;
        int sequenceLen = seedLen + totalSteps;

        // Create workspace sized for the seed plus the generated continuation.
        using var workspace = new Workspace(
            Workspace.CalculateRequiredSize(
                model.Config.HiddenDim, model.Config.FfnDim,
                model.Config.VocabSize, model.Config.NumLayers,
                sequenceLen));

        // Create independent KV caches so the model's own cache is untouched.
        var caches = new IKVCache[model.Config.NumLayers];
        try
        {
            for (int i = 0; i < caches.Length; i++)
                caches[i] = model.Config.IsShortConvLayer(i)
                    ? new ShortConvCache(model.Config.ShortConvCacheLength - 1, model.Config.HiddenDim, 1, sequenceLen)
                    : model.Config.IsRecurrentLayer(i)
                        ? new DeltaNetCache(model.Config.SsmConvKernel, model.Config.SsmConvDim,
                            model.Config.SsmTimeStepRank, model.Config.SsmStateSize,
                            model.Config.SsmValueHeadDim, 1, sequenceLen)
                        : new KVCache(1, model.Config.NumKvHeads, sequenceLen, model.Config.HeadDimForLayer(i));

            // Storage: one hidden vector per sample, K target token IDs per sample.
            using var hiddenStates = new Tensor<float>(numSamples, _hiddenDim);
            var targetTokens = new int[numSamples][];
            for (int i = 0; i < numSamples; i++)
                targetTokens[i] = new int[K];

            var seed = new int[seedLen];
            if (seedTokens is { Count: > 0 })
                for (int i = 0; i < seedLen; i++) seed[i] = seedTokens[i];
            else
                seed[0] = 0;

            static int ArgMax(ReadOnlySpan<float> values)
            {
                int best = 0;
                for (int i = 1; i < values.Length; i++)
                    if (values[i] > values[best]) best = i;
                return best;
            }

            // Greedy continuation after the seed. This fixes the token sequence
            // and therefore the K-head targets; hidden states are collected in
            // a second pass so they match generation exactly.
            var generated = new int[totalSteps];
            cancellationToken.ThrowIfCancellationRequested();
            workspace.Reset();
            using (var seedInput = workspace.Rent<int>([1, seedLen]))
            {
                for (int i = 0; i < seedLen; i++) seedInput.Data[i] = seed[i];
                using var seedLogits = model.ForwardLastLogits(seedInput, caches, 0, workspace);
                generated[0] = ArgMax(seedLogits.Data[.._vocabSize]);
            }
            progress?.Report(new MedusaCalibrationProgress(
                MedusaCalibrationPhase.Collecting, 1, totalSteps, null));

            for (int step = 1; step < totalSteps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                workspace.Reset();
                using var input = workspace.Rent<int>([1, 1]);
                input.Data[0] = generated[step - 1];

                using var logits = model.ForwardLastLogits(
                    input, caches, seedLen + step - 1, workspace);
                generated[step] = ArgMax(logits.Data[.._vocabSize]);
                progress?.Report(new MedusaCalibrationProgress(
                    MedusaCalibrationPhase.Collecting, step + 1, totalSteps, null));
            }

            // Replay the full seed + continuation through the blocks in one pass.
            // ForwardFill leaves the pre-norm hidden state; applying the final
            // norm here matches MedusaGenerator.RefillNormedHidden, which is
            // what the heads see while drafting.
            cancellationToken.ThrowIfCancellationRequested();
            for (int i = 0; i < caches.Length; i++) caches[i].Reset();
            workspace.Reset();
            using (var replay = new Tensor<int>(1, sequenceLen))
            {
                for (int i = 0; i < seedLen; i++) replay.Data[i] = seed[i];
                for (int step = 0; step < totalSteps; step++)
                    replay.Data[seedLen + step] = generated[step];
                model.ForwardFill(replay, caches, 0, workspace);
            }

            for (int step = 0; step < numSamples; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var row = model.GetNormedHiddenRow(seedLen - 1 + step)
                    ?? throw new InvalidOperationException("Medusa calibration lost the hidden state.");
                row.Data[.._hiddenDim].CopyTo(hiddenStates.RowSpan(step));
            }

            // Fill target tokens. At position P the LM head already supplies
            // offset 1 (token0), so head i must predict offset i+2:
            // generated[P + i + 1].
            for (int P = 0; P < numSamples; P++)
                for (int i = 0; i < K; i++)
                    targetTokens[P][i] = generated[P + i + 1];

            // Train the heads on the collected data.
            float finalLoss = Calibrate(hiddenStates, targetTokens, learningRate, trainingSteps,
                cancellationToken, progress);

            var result = new MedusaCalibrationResult(numSamples, trainingSteps, finalLoss);
            LastCalibration = result;
            return result;
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
