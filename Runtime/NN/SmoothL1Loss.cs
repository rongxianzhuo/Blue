using System;
using Blue.Core;
using Blue.Graph;

namespace Blue.NN
{
    public class SmoothL1Loss : IDisposable
    {

        public readonly ComputationalNode Output;
        public readonly Tensor Target;

        private readonly bool _isInnerTarget;
        private readonly Operate _backward;

        public static float Value => throw new NotImplementedException();

        public SmoothL1Loss(ComputationalNode output, Tensor target=null, float beta = 1.0f, float scale = 1.0f)
        {
            Output = output;
            _isInnerTarget = target == null;
            Target = target ?? new Tensor(output.Size);
            var strideOrder = output.Gradient.CalculateStrideOrder();
            _backward = new Operate("LossFunction/SmoothL1Loss", "CSMain")
                .SetInt("n", output.FlattenSize)
                .SetInt("dim", output.Size.Length)
                .SetFloat("beta", beta)
                .SetFloat("scale", scale)
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