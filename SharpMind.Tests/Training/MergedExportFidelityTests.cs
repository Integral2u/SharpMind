using System.Text;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Training;
using SharpMind.Training.LoRA;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// C4: a base that arrived as GGUF carries its tokenizer and <c>chat_template</c>
/// through the GGUF→SMM conversion and then through a LoRA fine-tune and merged
/// re-export, at both float targets. The point is that the fine-tune is a
/// weight-only operation: the served container keeps everything that made the
/// base usable as a chat model.
/// </summary>
public sealed class MergedExportFidelityTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private const int Vocab = 64;
    private const string ChatTemplate =
        "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] }}{% endfor %}{{ '<|assistant|>' }}";

    [Theory]
    [InlineData(QuantDType.F16)]
    [InlineData(QuantDType.Q8_0)]
    public void MergedExportFromGgufBase_KeepsTokenizerAndChatTemplate(QuantDType dtype)
    {
        string ggufPath = Path.Combine(_dir.Path, "base.gguf");
        WriteTinyGguf(ggufPath, BuildTokens(Vocab), ChatTemplate);

        // The base a user downloaded is GGUF; converting it to .SMM is what makes
        // it loadable by the training pipeline, and it must keep the metadata.
        string baseSmm = Path.Combine(_dir.Path, "base.smm");
        GgufToSmmConverter.Convert(ggufPath, baseSmm);
        Assert.Equal(ChatTemplate, SmmTrainingPipeline.LoadChatTemplate(baseSmm));

        using var model = SmmTrainingPipeline.LoadForTraining(
            baseSmm, new TrainingLoadOptions { Hardware = HardwareTier.Scalar },
            out var baseTokenizer, out var config);
        Assert.Equal("gpt2", config.Architecture);
        Assert.Equal(4, baseTokenizer.TokenToId("king"));
        Assert.Equal("king", baseTokenizer.IdToToken(4));

        // Fine-tune (adapters perturbed, then folded in) and re-export at the target dtype.
        using var lora = new LoRAModel(model, new LoRAConfig { Rank = 2, Alpha = 4f }, seed: 7);
        var rng = new Random(3);
        foreach (var p in lora.LoRAParameters())
            for (int i = 0; i < p.Data.ElementCount; i++)
                p.Data.Data[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        lora.Merge();

        string merged = Path.Combine(_dir.Path, $"merged-{dtype}.smm");
        SmmTrainingExporter.Export(model.Weights, baseTokenizer, merged, new SmmWriteOptions
        {
            QuantizationLevel = dtype,
            Source = "training",
        }, chatTemplate: ChatTemplate);

        // Tokenizer and chat_template survive the fine-tune + quantized re-export ...
        Assert.Equal(ChatTemplate, SmmTrainingPipeline.LoadChatTemplate(merged));
        using var reloaded = SmmTrainingPipeline.LoadForInference(merged, out var reloadedTokenizer, out var reloadedConfig);
        Assert.Equal(Vocab, reloadedConfig.VocabSize);
        Assert.Equal(baseTokenizer.VocabSize, reloadedTokenizer.VocabSize);
        Assert.Equal(baseTokenizer.TokenToId("king"), reloadedTokenizer.TokenToId("king"));
        Assert.Equal(baseTokenizer.IdToToken(4), reloadedTokenizer.IdToToken(4));

        // ... and the merged model still generates.
        var generated = SmmTrainingPipeline.GenerateGreedy(reloaded, [1, 4, 5], Vocab, steps: 3);
        Assert.Equal(6, generated.Count);
        Assert.All(generated, id => Assert.InRange(id, 0, Vocab - 1));
    }

    // ── Tiny GGUF writer (a GPT-2 tensor subset; mirrors what llama.cpp emits) ──

    private static string[] BuildTokens(int vocab)
    {
        string[] baseTokens = ["<unk>", "<s>", "</s>", "<pad>", "king", "queen", "dog", "cat"];
        var tokens = new string[vocab];
        for (int i = 0; i < vocab; i++)
            tokens[i] = i < baseTokens.Length ? baseTokens[i] : $"<t{i}>";
        return tokens;
    }

    private static void WriteTinyGguf(string path, string[] tokens, string chatTemplate)
    {
        int hidden = 32;
        using var fs = new FileStream(path, FileMode.Create);
        using var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

        w.Write(0x46554747u); // "GGUF"
        w.Write(3u);          // version
        w.Write(3L);          // tensor_count
        w.Write(13L);         // kv_count

        void KvString(string key, string value) { WriteGgufString(w, key); w.Write(8u); WriteGgufString(w, value); }
        void KvU32(string key, uint value) { WriteGgufString(w, key); w.Write(4u); w.Write(value); }
        void KvStringArray(string key, string[] values)
        {
            WriteGgufString(w, key);
            w.Write(9u);
            w.Write(8u);
            w.Write((ulong)values.Length);
            foreach (string v in values) WriteGgufString(w, v);
        }

        KvString("general.architecture", "gpt2");
        KvU32("gpt2.embedding_length", (uint)hidden);
        KvU32("gpt2.block_count", 1);
        KvU32("gpt2.feed_forward_length", 64);
        KvU32("gpt2.context_length", 16);
        KvU32("gpt2.attention.head_count", 4);
        KvU32("gpt2.attention.head_count_kv", 4);
        KvString("tokenizer.ggml.model", "gpt2");
        KvString("tokenizer.ggml.pre", "qwen2");
        KvStringArray("tokenizer.ggml.tokens", tokens);
        KvU32("tokenizer.ggml.bos_token_id", 1);
        KvU32("tokenizer.ggml.eos_token_id", 2);
        KvString("tokenizer.chat_template", chatTemplate);

        byte[] emb = MakeFloatData(tokens.Length * hidden, seed: 7);
        byte[] norm = MakeFloatData(hidden, seed: 11);
        byte[] attn = MakeFloatData(hidden * hidden, seed: 13);

        void TensorInfo(string name, int[] shape, long offset)
        {
            WriteGgufString(w, name);
            w.Write((uint)shape.Length);
            foreach (int d in shape) w.Write((ulong)d);
            w.Write(0u); // F32
            w.Write((ulong)offset);
        }

        TensorInfo("token_embd.weight", [tokens.Length, hidden], 0);
        TensorInfo("output_norm.weight", [hidden], emb.Length);
        TensorInfo("blk.0.attn_q.weight", [hidden, hidden], emb.Length + norm.Length);

        long end = fs.Position;
        long aligned = (end + 31) & ~31L;
        if (aligned > end) fs.Write(new byte[aligned - end]);

        fs.Write(emb, 0, emb.Length);
        fs.Write(norm, 0, norm.Length);
        fs.Write(attn, 0, attn.Length);
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static byte[] MakeFloatData(int count, int seed)
    {
        var rng = new Random(seed);
        var bytes = new byte[count * 4];
        for (int i = 0; i < count; i++)
            BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), (float)(rng.NextDouble() * 2 - 1));
        return bytes;
    }
}
