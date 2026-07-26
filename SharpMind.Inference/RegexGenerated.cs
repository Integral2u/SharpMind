using System.Text.RegularExpressions;

namespace SharpMind.Inference
{
    public partial class RegexGenerated
    {
        [GeneratedRegex(@"<\|[^|]+\|>")]
        public static partial Regex ChatMLTokens { get; }
        [GeneratedRegex(@"^for\s+(\w+)\s+in\s+(.+)$")]
        public static partial Regex JinjaForVarInExpr { get; }

        [GeneratedRegex(@"^set\s+(\w+)\s*=\s*namespace\s*\((.*)?\)\s*$", RegexOptions.Singleline)]
        public static partial Regex JinjaSetNsFieldValue { get; }
        [GeneratedRegex(@"(\w+)\s*=\s*([^,]+)")]
        public static partial Regex JinjaNamespace { get; }
        [GeneratedRegex(@"^set\s+(\w+)\.(\w+)\s*=\s*(.+)$")]
        public static partial Regex JinjaNamespaceDotFieldEqExpr { get; }
        [GeneratedRegex(@"^set\s+(\w+)\s*=\s*(.+)$", RegexOptions.Singleline)]
        public static partial Regex JinjaSetVarEqExpr { get; }
        [GeneratedRegex(@"^not\s+\w+\s+is\s+defined$")]
        public static partial Regex JinjaNotXIsDefined { get; }
        [GeneratedRegex(@"^(\w+)\s+is\s+defined$")]
        public static partial Regex JinjaXIsDefined { get; }
        [GeneratedRegex(@"^(.+?)\s+is\s+(not\s+)?none$")]
        public static partial Regex JinjaXIsNoneNotNone { get; }
        [GeneratedRegex(@"not\s+(\w+)")]
        public static partial Regex JinjaNotX { get; }
        [GeneratedRegex(@"^(\w+)")]
        public static partial Regex JinjaIsX { get; }
        [GeneratedRegex(@"^'([^']*)'\s+in\s+(.+)$")]
        public static partial Regex JinjaLiteralInXSubstr { get; }
        [GeneratedRegex("^\"([^\"]*)\"\\s+in\\s+(.+)$")]
        public static partial Regex JinjaLiteralInX { get; }
        [GeneratedRegex(@"^(.+?)\s*\|\s*trim\s*$")]
        public static partial Regex JinjaExprTrim { get; }
        [GeneratedRegex(@"^(.+?)\s*\|\s*length\s*$")]
        public static partial Regex JinjaExprLength { get; }
        [GeneratedRegex(@"^(.+?)\s+is\s+(not\s+)?defined$")]
        public static partial Regex JinjaIsDefined { get; }
        [GeneratedRegex(@"^(\w+)\.split\('([^']*)'\)\[(-?\d+)\]$")]
        public static partial Regex JinjaSplitDelim { get; }
        [GeneratedRegex(@"^(\w+)\.(\w+)$")]
        public static partial Regex JinjaObjDotField { get; }
        [GeneratedRegex(@"^(.+?)\s*([+\-])\s*(\d+)$")]
        public static partial Regex JinjaPlusMinusN { get; }
        [GeneratedRegex(@"^(.+?)\s+is\s+(not\s+)?(true|false)$")]
        public static partial Regex JinjaIsBool { get; }
        [GeneratedRegex(@"^(.+?)\s+is\s+(not\s+)?string$")]
        public static partial Regex JinjaIsString { get; }
        [GeneratedRegex(@"^(.+?)\s+is\s+(not\s+)?iterable$")]
        public static partial Regex JinjaIsIterable { get; }
    }
}
