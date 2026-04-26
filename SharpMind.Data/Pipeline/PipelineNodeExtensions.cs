namespace SharpMind.Data.Pipeline;
internal static class PipelineNodeExtensions
{
    internal static string Indent(int depth) => new(' ', depth * 2);
}