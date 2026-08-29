using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>One thread per row. // ponytail: a serial row of V=151936; warp-reduce is the upgrade.</summary>
internal static class LossKernels
{
    // rowLoss[t] = −[(1−ε)·logp[label] + ε/V·Σ logp]   (CrossEntropyLoss.Compute) ; logits[t,:] ← (softmax − u)·scale   (Gradients.CrossEntropySoftmax)
    public static void CeRow(Index1D t, ArrayView<float> logits, ArrayView<int> labels, ArrayView<float> rowLoss, int v, int ignoreId, float smooth, float scale)
    {
        long b = (long)(int)t * v;
        int label = labels[t];
        if (label == ignoreId)
        {
            rowLoss[t] = 0f;
            for (int i = 0; i < v; i++) logits[b + i] = 0f;
            return;
        }
        // ponytail: sumExp/sumRow accumulate in float across V=151936 terms at production shape —
        // straight per-thread accumulation, no Kahan/pairwise compensation. OpenCL on this device
        // (gfx1152) has no reliable double support to fall back to; the CPU oracle in
        // LossKernelTests accumulates in float too (CrossEntropyLoss.Compute), so this is parity,
        // not a regression. Upgrade to a tree/pairwise reduction if V grows enough to show drift.
        float max = logits[b]; float sumRow = logits[b];
        for (int i = 1; i < v; i++) { float x = logits[b + i]; if (x > max) max = x; sumRow += x; }
        float sumExp = 0f;
        for (int i = 0; i < v; i++) sumExp += XMath.Exp(logits[b + i] - max);
        float logSum = XMath.Log(sumExp);
        float logProbLabel = (logits[b + label] - max) - logSum;
        if (smooth <= 0f) rowLoss[t] = -logProbLabel;
        else rowLoss[t] = -((1f - smooth) * logProbLabel + (smooth / v) * (sumRow - v * max - v * logSum));

        float inv = 1f / sumExp; float epsOverV = smooth / v;
        for (int i = 0; i < v; i++)
        {
            float p = XMath.Exp(logits[b + i] - max) * inv;
            if (smooth > 0f) p -= epsOverV;
            logits[b + i] = p * scale;
        }
        logits[b + label] -= (smooth > 0f ? 1f - smooth : 1f) * scale;
    }
}
