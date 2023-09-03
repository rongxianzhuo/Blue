using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class TensorNode : IGraphNode
    {

        public readonly Tensor TotalGradient;
        public readonly int Id;

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public bool IsParameter => TotalGradient != null;

        public TensorNode(int id, int size, bool isParam)
        {
            Id = id;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
        }

        public TensorNode(int id, bool isParam, List<float> data)
        {
            Id = id;
            var size = data.Count;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(data);
            _gradient = new Tensor(size);
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
            if (TotalGradient == null) return;
            Op.Increment(TotalGradient, _gradient);
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            TotalGradient?.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
        }
    }
}