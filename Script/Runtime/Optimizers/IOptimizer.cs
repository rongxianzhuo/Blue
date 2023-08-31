using Blue.Core;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void Step(Tensor param, Tensor gradient);

        void Destroy();
    }
}