using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace SharpMind.Model
{
    public partial class RegexGenerated
    {
        [GeneratedRegex(@"<[^>]+>")]
        public static partial Regex ChatTemplateRegex { get; }
        [GeneratedRegex(@"\.(\d+)\.")]
        public static partial Regex LayerIndexDot7Regex { get; }
        [GeneratedRegex(@"blk\.(\d+)")]
        public static partial Regex LayerIndexBlkDot7Regex { get; }
        [GeneratedRegex(@"layer_(\d+)")]
        public static partial Regex LayerIndexLayerDot7Regex { get; }

    }
}
