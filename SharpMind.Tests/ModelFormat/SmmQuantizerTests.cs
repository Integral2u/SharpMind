using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using System.IO;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Verifies <see cref="SmmQuantizer"/>: re-quantizing an F32 .SMM container in
/// place must change the affected tensors' dtypes, shrink the data region
/// (for the 256-block K-quants) and keep the recovered weights within the
/// encoders' fidelity windows. Non-256-multiple tensors must softly fall back
/// to F16 instead of failing the whole model.
/// </summary>
public class SmmQuantizerTests : IDisposable
{
    private readonly TempDirectory _temp = new();
    public void Dispose() => _temp.Dispose();

    private static ModelConfig Config => new()
    {
        VocabSize = 32,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 8,
        NormEps = 1e-3f,
    };

    private static SmmTensorData Tensor(string name, int count, int seed, int[]? shape = null)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = (float)(rng.NextDouble() * 2 - 1);
        var bytes = new byte[count * 4];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return new SmmTensorData
        {
            Name = name,
            Shape = shape ?? [count],
            Dtype = QuantDType.F32,
            GetBytes = () => bytes,
        };
    }

    /// <summary>
    /// A model with two K-quant-eligible tensors (256-aligned) plus one small
    /// norm tensor whose flattened length is not a multiple of 256.
    /// </summary>
    private string WriteSmm(string name)
    {
        string path = Path.Combine(_temp.Path, name);
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Tensor("token_embd.weight", 256, seed: 1, shape: [32, 8]),
            Tensor("blk.0.attn_q.weight", 256, seed: 2, shape: [32, 8]),
            Tensor("output_norm.weight", 16, seed: 3),
        ]);
        return path;
    }

    public static IEnumerable<object[]> QuantLevels()
    {
        yield return [QuantDType.Q2_K];
        yield return [QuantDType.Q4_K];
        yield return [QuantDType.Q6_K];
        yield return [QuantDType.Q8_K];
    }

    /// <summary>
    /// A Q8_0 tensor pre-encoded from floats, mimicking what the GGUF converter
    /// produces for an already-quantized source model.
    /// </summary>
    private static SmmTensorData Q8_0Tensor(string name, int count, int seed, int[]? shape = null)
    {
        var rng = new Random(seed);
        var floats = new float[count];
        for (int i = 0; i < count; i++) floats[i] = (float)(rng.NextDouble() * 2 - 1);
        var raw = TensorQuantizer.Quantize(floats, shape ?? [count], QuantDType.Q8_0);
        return new SmmTensorData
        {
            Name = name,
            Shape = shape ?? [count],
            Dtype = QuantDType.Q8_0,
            GetBytes = () => raw,
        };
    }

    private string WriteQ8_0Smm(string name)
    {
        string path = Path.Combine(_temp.Path, name);
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Q8_0Tensor("token_embd.weight", 256, seed: 1),
            Q8_0Tensor("blk.0.attn_q.weight", 256, seed: 2),
            Q8_0Tensor("output_norm.weight", 32, seed: 3),
        ]);
        return path;
    }

    [Theory]
    [InlineData(QuantDType.Q4_K)]
    [InlineData(QuantDType.Q6_K)]
    public void Quantize_AlreadyQuantizedSource_RequantizesAndShrinks(QuantDType target)
    {
        // A model that is ALREADY Q8_0 (e.g. converted from a Q8_0 GGUF) must
        // still be requantizable to a leaner K-quant — this was the "zero bytes
        // saved" path where only F32 tensors were touched.
        string path = WriteQ8_0Smm("requant.smm");
        long before = new FileInfo(path).Length;

        SmmQuantizer.Quantize(path, target);

        var entries = SmmLoader.ReadTensorIndex(path);
        var byName = entries.ToDictionary(e => e.Name, e => e.Dtype);
        Assert.Equal(target, byName["token_embd.weight"]);
        Assert.Equal(target, byName["blk.0.attn_q.weight"]);
        // 16 elements: neither Q4_K nor F16 is leaner than the 34-byte Q8_0
        // row-batched block, so the norm passes through untouched.
        Assert.Equal(QuantDType.Q8_0, byName["output_norm.weight"]);
        Assert.True(new FileInfo(path).Length < before, "re-quantized container should be smaller than Q8_0");
    }

    [Theory]
    [InlineData(QuantDType.Q4_K)]
    [InlineData(QuantDType.Q6_K)]
    public void Quantize_SaveAs_LeavesSourceUntouched(QuantDType target)
    {
        string source = WriteSmm("saveas-source.smm");
        byte[] originalSource = File.ReadAllBytes(source);
        string dest = Path.Combine(_temp.Path, "saveas-dest.smm");

        SmmQuantizer.Quantize(source, dest, target);

        Assert.Equal(originalSource, File.ReadAllBytes(source)); // source byte-identical
        Assert.True(File.Exists(dest));
        var byName = SmmLoader.ReadTensorIndex(dest).ToDictionary(e => e.Name, e => e.Dtype);
        Assert.Equal(target, byName["token_embd.weight"]);
        Assert.True(new FileInfo(dest).Length < new FileInfo(source).Length);
    }

    [Fact]
    public void Quantize_SaveAs_CoarserThanSource_IsPassthrough()
    {
        // Re-encoding an already-coarse tensor to a FINER dtype would only add
        // loss for no size benefit — the quantizer must copy it verbatim instead
        // of degrading it ("never upscale"). Q8_0 of 256 elems is 272 B vs Q8_K's
        // 292 B, so Q8_K is genuinely not leaner here.
        string source = WriteQ8_0Smm("upscale-source.smm");
        string dest = Path.Combine(_temp.Path, "upscale-dest.smm");

        SmmQuantizer.Quantize(source, dest, QuantDType.Q8_K);

        Assert.Equal(QuantDType.Q8_0, SmmLoader.ReadTensorIndex(dest)[0].Dtype);
        Assert.Equal(new FileInfo(source).Length, new FileInfo(dest).Length);
    }

    [Theory]
    [MemberData(nameof(QuantLevels))]
    public void Quantize_ChangesDtypesAndShrinksFile(QuantDType target)
    {
        string path = WriteSmm("quant.smm");
        long before = new FileInfo(path).Length;

        SmmQuantizer.Quantize(path, target);

        var entries = SmmLoader.ReadTensorIndex(path);
        var byName = entries.ToDictionary(e => e.Name, e => e.Dtype);
        Assert.Equal(target, byName["token_embd.weight"]);
        Assert.Equal(target, byName["blk.0.attn_q.weight"]);
        Assert.Equal(QuantDType.F16, byName["output_norm.weight"]); // not 256-aligned → F16 fallback
        Assert.True(new FileInfo(path).Length < before, "quantized container should be smaller than F32");
    }

    [Theory]
    [MemberData(nameof(QuantLevels))]
    public void Quantize_RoundTripsWithinTolerance(QuantDType target)
    {
        string path = WriteSmm("roundtrip.smm");
        var original = SnapshotValues(path);

        SmmQuantizer.Quantize(path, target);

        var entries = SmmLoader.ReadTensorIndex(path);
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        foreach (var entry in entries)
        {
            long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
            byte[] raw = SmmLoader.ReadTensorBytes(path, entry, rawSize);
            using var ms = new MemoryStream(raw);
            using var reader = new BinaryReader(ms);

            int count = 1;
            foreach (int d in entry.Shape) count *= d;
            var read = new float[count];
            qOps.ReadFor(entry.Dtype, reader, read, count);

            float[] reference = original[entry.Name];
            double errSq = 0, sigSq = 0;
            for (int i = 0; i < count; i++)
            {
                float e = reference[i] - read[i];
                errSq += e * e;
                sigSq += (double)reference[i] * reference[i];
            }
            double rel = Math.Sqrt(errSq / Math.Max(1e-30f, sigSq));

            if (entry.Dtype == QuantDType.F16)
                Assert.True(rel < 1e-2, $"{entry.Name}: {rel:P3} exceeds F16 tolerance");
            else
                Assert.True(rel < MaxRelError(target),
                    $"{target} {entry.Name}: relative weight RMS error {rel:P3} exceeds tolerance");
        }
    }

    private static float MaxRelError(QuantDType dtype) => dtype switch
    {
        QuantDType.Q8_K => 0.010f,
        QuantDType.Q6_K => 0.040f,
        QuantDType.Q5_K => 0.080f,
        QuantDType.Q4_K => 0.120f,
        QuantDType.Q3_K => 0.250f,
        QuantDType.Q2_K => 0.350f,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    private static Dictionary<string, float[]> SnapshotValues(string path)
    {
        var result = new Dictionary<string, float[]>();
        var entries = SmmLoader.ReadTensorIndex(path);
        foreach (var entry in entries)
        {
            long rawSize = QuantizationOps.GetRawTensorByteCount(entry.Shape, entry.Dtype);
            byte[] raw = SmmLoader.ReadTensorBytes(path, entry, rawSize);
            int count = 1;
            foreach (int d in entry.Shape) count *= d;
            var values = new float[count];
            Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
            result[entry.Name] = values;
        }
        return result;
    }

    [Fact]
    public void Quantize_PreservesMetaTokenizerAndPlugins()
    {
        string path = WriteSmm("meta.smm");
        var doc = SmmModifier.Read(path);
        doc.SystemPrompt = "Quantize me.";
        SmmModifier.Write(path, doc);

        SmmQuantizer.Quantize(path, QuantDType.Q8_K);

        Assert.Equal("Quantize me.", SmmLoader.LoadSystemPrompt(path));
        Assert.True(new FileInfo(path).Length > 0);
    }

    [Fact]
    public void Quantize_RejectsUnsupportedTarget()
    {
        string path = WriteSmm("reject.smm");
        Assert.Throws<NotSupportedException>(() => SmmQuantizer.Quantize(path, QuantDType.I8));
        Assert.False(File.Exists(path + ".tmp"));
    }

    [Fact]
    public void Quantize_MissingFile_Throws()
    {
        Assert.Throws<FileNotFoundException>(() => SmmQuantizer.Quantize(
            Path.Combine(_temp.Path, "missing.smm"), QuantDType.Q8_K));
    }

    // ── Options / planner ───────────────────────────────────────────────────

    [Fact]
    public void Plan_ManualPerRole_UsesOverridesAndSafeF16Defaults()
    {
        string path = WriteSmm("plan-manual.smm");
        var entries = SmmLoader.ReadTensorIndex(path);

        var plan = SmmQuantPlan.Resolve(entries, new SmmQuantOptions
        {
            DefaultLevel = QuantDType.Q8_K,
            RoleLevels = new Dictionary<SmmTensorRole, QuantDType>
            {
                [SmmTensorRole.Ffn] = QuantDType.Q6_K,
            },
        });

        // attn_q → Attention → not overridden → DefaultLevel
        Assert.Equal(QuantDType.Q8_K, plan["blk.0.attn_q.weight"]);
        // output_norm → Norm → not quantizable → F16
        Assert.Equal(QuantDType.F16, plan["output_norm.weight"]);
    }

    [Fact]
    public void ResolveRole_ClassifiesByNamingConventions()
    {
        Assert.Equal(SmmTensorRole.Embedding, SmmQuantPlan.ResolveRole("token_embd.weight", false));
        Assert.Equal(SmmTensorRole.Attention, SmmQuantPlan.ResolveRole("blk.0.attn_q.weight", false));
        Assert.Equal(SmmTensorRole.Attention, SmmQuantPlan.ResolveRole("blk.0.q_proj.weight", false));
        Assert.Equal(SmmTensorRole.Ffn, SmmQuantPlan.ResolveRole("blk.0.ffn_up.weight", false));
        Assert.Equal(SmmTensorRole.Norm, SmmQuantPlan.ResolveRole("blk.0.ffn_norm.weight", false));
        Assert.Equal(SmmTensorRole.Bias, SmmQuantPlan.ResolveRole("blk.0.attn_q.bias", false));
        Assert.Equal(SmmTensorRole.LmHead, SmmQuantPlan.ResolveRole("output.weight", false));
        Assert.Equal(SmmTensorRole.Expert, SmmQuantPlan.ResolveRole("blk.0.exps.3.ffn_up.weight", true));
        Assert.Equal(SmmTensorRole.Router, SmmQuantPlan.ResolveRole("blk.0.ffn_gate.weight", true));
        Assert.Equal(SmmTensorRole.Unknown, SmmQuantPlan.ResolveRole("unexpected.tensor", false));
    }

    [Fact]
    public void Plan_Budget_CoarsensRolesUntilItFits()
    {
        // A model with one big attention tensor and one small norm.
        string path = Path.Combine(_temp.Path, "budget.smm");
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Tensor("blk.0.attn_q.weight", 4096, seed: 1, shape: [256, 16]),
        ]);
        long f32Size = new FileInfo(path).Length;

        // Target impossible at the default fine level — force the planner down.
        long budget = (long)(f32Size * 0.3);
        var plan = SmmQuantPlan.Resolve(SmmLoader.ReadTensorIndex(path), new SmmQuantOptions
        {
            TargetBytes = budget,
            Floor = QuantDType.Q2_K,
        }, f32Size);

        Assert.True(plan.ContainsKey("blk.0.attn_q.weight"));
        var dtype = plan["blk.0.attn_q.weight"];
        long quantized = QuantizationOps.GetRawTensorByteCount([256, 16], dtype);
        long meta = f32Size - 4096 * 4;
        Assert.True(meta + quantized <= budget,
            $"{dtype}: est. {meta + quantized:N0}B exceeds budget {budget:N0}B");
    }

    [Fact]
    public void Plan_Budget_ThrowsWhenFloorImpossible()
    {
        string path = Path.Combine(_temp.Path, "budget-impossible.smm");
        SmmWriter.Write(path, Config, tokenizer: null, chatTemplate: null, tensors:
        [
            Tensor("blk.0.attn_q.weight", 256, seed: 1, shape: [256]),
        ]);
        long f32Size = new FileInfo(path).Length;
        var entries = SmmLoader.ReadTensorIndex(path);

        // A hopeless budget at the coarsest floor (Q2_K) must fail loudly.
        Assert.Throws<InvalidOperationException>(() => SmmQuantPlan.Resolve(entries, new SmmQuantOptions
        {
            TargetBytes = 100,
            Floor = QuantDType.Q2_K,
        }, f32Size));
    }
}