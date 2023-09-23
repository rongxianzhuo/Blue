using System;
using Blue.Core;

namespace Blue.Graph
{
    public class TensorNode : IGraphNode
    {

        public readonly int Id;

        public readonly bool IsParameter;

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public TensorNode(int id, bool isParam, params int[] size)
        {
            Id = id;
            IsParameter = isParam;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
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
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
        }
    }
}