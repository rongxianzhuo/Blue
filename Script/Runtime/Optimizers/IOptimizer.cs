using Blue.Graph;
using UnityEngine;

namespace Blue.Optimizers
{
    public interface IOptimizer
    {
        void OnBackwardPropagation(ComputeBuffer param, ComputeBuffer gradient);

        void Destroy();
    }
}