using Blue.Operates;
using UnityEngine;

namespace Blue.Optimizers
{
    public class DummyOptimizer : IOptimizer
    {
        
        public void OnBackwardPropagation(ComputeBuffer param, ComputeBuffer gradient)
        {
            AddOperate.Calculate(param, gradient, -0.01f, 0);
        }

        public void Destroy()
        {
            
        }
    }
}