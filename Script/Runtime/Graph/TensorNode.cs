using System;
using Blue.Core;

namespace Blue.Graph
{
    public class TensorNode : IGraphNode
    {

        public readonly int Id;

        public readonly Tensor TotalGradient;

        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly Operate _increase;

        public bool IsParameter => TotalGradient != null;

        public TensorNode(int id, bool isParam, params int[] size)
        {
            Id = id;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
            if (isParam)
            {
                _increase = new Operate("Common/GradientIncrease", "CSMain")
                    .SetFloat("weight_decay", 0.000f)
                    .SetTensor("gradient", _gradient)
                    .SetTensor("weight", _output)
                    .SetTensor("total_gradient", TotalGradient)
                    .SetDispatchSize(TotalGradient.FlattenSize);
            }
            else
            {
                _increase = null;
            }
        }

        public void Resize(params int[] size)
        {
            _output.Resize(size);
            _gradient.Resize(size);
        }
        
        public Tensor GetOutput()
        {
            return _output;
        }

        public Tensor GetGradient()
        {
            return _gradient;
        }

        public void Forward()
        {
        }

        public void Backward()
        {
            _increase?.Dispatch();
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            TotalGradient?.Release();
            _increase?.Destroy();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
        }
    }
}