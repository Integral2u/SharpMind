using System.Text.RegularExpressions;

namespace SharpMind.Model
{
    public partial class RegexGenerated
    {
        [GeneratedRegex(@"blk\.(\d+)\.")]
        public static partial Regex LayerIndexDotNDot { get; }
        [GeneratedRegex(@"<[^>]+>")]
        public static partial Regex ChatTemplateRegex { get; }
        [GeneratedRegex(@"\.(\d+)\.")]
        public static partial Regex LayerIndexDot7Regex { get; }
        [GeneratedRegex(@"blk\.(\d+)")]
        public static partial Regex LayerIndexBlkDot7Regex { get; }
        [GeneratedRegex(@"layer_(\d+)")]
        public static partial Regex LayerIndexLayerDot7Regex { get; }
        // Per-expert index; group 1 is the expert id. Three spellings for the same thing:
//   ffn_gate.exps.3.weight   GGUF, llama.cpp's ".exps." form (Qwen2-MoE)
//   blk.0.exps.3.ffn_gate    .smm, index *before* the role rather than after
//   ffn_gate.3.weight         GGUF, bare dotted index (Mixtral)
// Anchoring only to the role missed .smm; anchoring to a bare "\.(\d+)\." was worse,
// because it also matched the *layer* index and filed blk.0.ffn_gate_inp (the router)
// under expert 0. The index is therefore only recognised directly beside ".exps." or
// an "ffn_{gate,up,down}." token, which leaves "ffn_gate_inp" alone.
[GeneratedRegex(@"(?:\.exps\.|ffn_(?:gate|up|down)\.)(\d+)")]
        public static partial Regex ExpertIndex { get; }

    }
}
