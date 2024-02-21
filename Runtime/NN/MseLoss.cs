using System;
using Blue.Core;
using Blue.Graph;

namespace Blue.Runtime.NN
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
                foreach (var f in Output.Gradient.Sync())
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
            _backward = new Operate("LossFunction/MseLoss", "CSMain")
                .SetInt("n", output.FlattenSize)
                .SetTensor("output", output)
                .SetTensor("target", Target)
                .SetTensor("gradient", output.Gradient)
                .SetDispatchSize(output.FlattenSize);
        }

        public void Backward()
        {
            _backward.Dispatch();
        }

        public void Dispose()
        {
            if (_isInnerTarget) Target.Dispose();
            _backward.Dispose();
        }
    }
}