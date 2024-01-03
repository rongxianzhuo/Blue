using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void Step(IReadOnlyCollection<ComputationalNode> node);

        void Destroy();
    }
}