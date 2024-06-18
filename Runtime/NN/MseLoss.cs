using System;
using Blue.Core;
using Blue.Graph;

namespace Blue.NN
{
    
    public class MseLoss : IDisposable
    {

        public readonly ComputationalNode Output;
        public readonly Tensor Target;

        private readonly bool _isInnerTarget;
        private readonly Operate _backward;

        public float Value
        {
            get
            {
                var sum = 0f;
                foreach (var f in Output.Gradient.Sync<float>())
                {
                    sum += f * f * Output.FlattenSize / 4;
                }
                return sum;
            }
        }
        
        public MseLoss(ComputationalNode output, Tensor target=null)
        {
            Output = output;
            _isInnerTarget = target == null;
            Target = target ?? new Tensor(output.Size);
            var strideOrder = output.Gradient.CalculateStrideOrder();
            _backward = new Operate("LossFunction/MseLoss", "CSMain")
                .SetInt("n", output.FlattenSize)
                .SetInt("dim", output.Size.Length)
                .SetTensor("output", output, strideOrder)
                .SetTensor("target", Target, strideOrder)
                .SetTensor("gradient", output.Gradient, strideOrder)
                .SetDispatchSize(output.FlattenSize);
        }

        public void Backward()
        {
            _backward.Dispatch();
            Output.Backward();
        }

        public void Dispose()
        {
            if (_isInnerTarget) Target.Dispose();
            _backward.Dispose();
        }
    }
}