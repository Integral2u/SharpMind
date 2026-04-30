using System;
using System.Runtime.CompilerServices;

namespace SharpMind.Data.Pipeline;
internal static class PipelineNodeExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static string Indent(int depth) => new(' ', depth * 2);
}