using System;
using Blue.Core;
using Blue.Kit;

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
            _increase = isParam ? Op.Increment(TotalGradient, _gradient) : null;
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