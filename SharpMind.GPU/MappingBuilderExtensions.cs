using SharpMind.Core;

namespace SharpMind.GPU
{
    public static class MappingBuilderExtensions
    {        
        [Obsolete("Inference has no GPU path anymore; this is a no-op. See GPU-accelerated training for the current GPU support.")]
        public static MappingBuilder WithGpu(this MappingBuilder builder, bool nonQuant = true, bool vecDot = false, bool matMul = false)
        {
            return builder;
        }
    }
}
