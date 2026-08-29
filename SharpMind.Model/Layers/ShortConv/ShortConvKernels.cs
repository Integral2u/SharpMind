using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers.ShortConv;

/// <summary>
/// Depthwise 1D convolution backing an LFM2 short-conv (no-attention) block.
///
/// Reference semantics (llama.cpp <c>ggml_compute_forward_ssm_conv_f32</c> with a
/// <c>d_conv - 1</c> row history pasted ahead of the current step's gated rows):
/// the conv runs over the chronological buffer <c>[state | bx]</c> where
/// <c>state</c> is the last <c>taps - 1</c> rows produced by the previous step and
/// <c>bx</c> are this step's rows. Output row <c>t</c> is the depthwise dot of the
/// kernel with buffer rows <c>[t, t + taps)</c>, so output <c>t</c> corresponds to
/// input token <c>t</c> and the first output of any step re-reads the retained
/// history. After the step the retained state is the last <c>taps - 1</c> rows of
/// the concatenated buffer.
///
/// A cold start simply has a zero-filled state: the first <c>taps - 1</c> outputs
/// of the very first prefill therefore see leading zeros, exactly like llama.cpp.
/// </summary>
internal static class ShortConvKernels
{
    /// <summary>
    /// Computes the shape-gated conv input <c>bx = b ⊙ x</c> into <paramref name="bx"/>
    /// from the in-projection output <paramref name="projected"/> [rows, 3H].
    /// </summary>
    internal static unsafe void ComputeGatedInput(
        Tensor<float> projected, Tensor<float> bx, int rows, int hidden)
    {
        fixed (float* p = projected.Data, pb = bx.Data)
        {
            int stride = 3 * hidden;
            for (int r = 0; r < rows; r++)
            {
                float* b = p + r * stride;
                float* x = b + 2 * hidden;
                float* o = pb + r * hidden;
                for (int c = 0; c < hidden; c++)
                    o[c] = b[c] * x[c];
            }
        }
    }

    /// <summary>
    /// Applies the output gate in place: <c>convOut = c ⊙ convOut</c>, where
    /// <c>c</c> is channels [H, 2H) of <paramref name="projected"/> [rows, 3H].
    /// </summary>
    internal static unsafe void ApplyOutputGate(
        Tensor<float> projected, Tensor<float> convOut, int rows, int hidden)
    {
        fixed (float* p = projected.Data, pOut = convOut.Data)
        {
            int stride = 3 * hidden;
            for (int r = 0; r < rows; r++)
            {
                float* c = p + r * stride + hidden;
                float* o = pOut + r * hidden;
                for (int h = 0; h < hidden; h++)
                    o[h] *= c[h];
            }
        }
    }

    /// <summary>
    /// Sliding-window depthwise conv over <c>[state | bx]</c> into
    /// <paramref name="output"/> [batch, seq, hidden].
    /// </summary>
    /// <param name="state">[batch, taps - 1, hidden] rows retained from the previous step.</param>
    /// <param name="kernel">[taps, hidden] F32 conv kernel, GGUF row-major [kernelRow, channel].</param>
    internal static unsafe void ApplyConv(
        Tensor<float> bx, Tensor<float> state, Tensor<float> kernel,
        Tensor<float> output, int batch, int seq, int hidden, int taps)
    {
        int stateRows = taps - 1;
        fixed (float* pBx = bx.Data, pState = state.Data, pKernel = kernel.Data, pOut = output.Data)
        {
            int bxStride = seq * hidden;
            int stStride = stateRows * hidden;
            for (int b = 0; b < batch; b++)
            {
                float* sbx = pBx + b * bxStride;
                float* sst = pState + b * stStride;
                float* so = pOut + b * bxStride;
                for (int t = 0; t < seq; t++)
                {
                    float* orow = so + t * hidden;
                    for (int c = 0; c < hidden; c++)
                    {
                        float acc = 0f;
                        for (int k = 0; k < taps; k++)
                        {
                            int r = t + k;
                            float v = r < stateRows
                                ? sst[r * hidden + c]
                                : sbx[(r - stateRows) * hidden + c];
                            acc += v * pKernel[k * hidden + c];
                        }
                        orow[c] = acc;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Rolls the retained state up to the last <c>taps - 1</c> rows of the
    /// <c>[state | bx]</c> chronology. Handles the mixed case where the step
    /// produced fewer than <c>taps - 1</c> new rows (a single-token decode).
    /// </summary>
    internal static unsafe void UpdateState(
        Tensor<float> bx, Tensor<float> state, int batch, int seq, int hidden, int stateRows)
    {
        fixed (float* pBx = bx.Data, pState = state.Data)
        {
            int bxStride = seq * hidden;
            int stStride = stateRows * hidden;
            for (int b = 0; b < batch; b++)
            {
                float* sbx = pBx + b * bxStride;
                float* sst = pState + b * stStride;
                for (int i = 0; i < stateRows; i++)
                {
                    int src = seq + i; // row index within the [state | bx] chronology
                    float* srcRow = src < stateRows
                        ? sst + src * hidden
                        : sbx + (src - stateRows) * hidden;
                    Buffer.MemoryCopy(srcRow, sst + i * hidden, (nuint)(hidden * sizeof(float)), (nuint)(hidden * sizeof(float)));
                }
            }
        }
    }
}