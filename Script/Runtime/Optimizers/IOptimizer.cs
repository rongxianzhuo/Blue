using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void Step(IGraphNode node);

        void Destroy();
    }
}