using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void Step(TensorNode node);

        void Destroy();
    }
}