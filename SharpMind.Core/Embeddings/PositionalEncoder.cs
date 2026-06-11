using SharpMind.Core.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind.Core.Embeddings
{
    public abstract class PositionalEncoder
    {
        public abstract void Apply(Tensor<float> x, int positionOffset = 0);
        public abstract void ApplyBatched(Tensor<float> x, int positionOffset = 0);
    }
}
