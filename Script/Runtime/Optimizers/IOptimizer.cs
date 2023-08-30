using UnityEngine;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void Step(ComputeBuffer param, ComputeBuffer gradient);

        void Destroy();
    }
}