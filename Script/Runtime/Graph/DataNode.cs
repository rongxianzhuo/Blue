using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    public class DataNode : IGraphNode
    {

        public readonly ComputeBuffer TotalGradient;

        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public DataNode(int size, bool isParam)
        {
            TotalGradient = isParam ? new ComputeBuffer(size, 4) : null;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }
        
        public ComputeBuffer GetOutput()
        {
            return _output;
        }

        public ComputeBuffer GetGradient()
        {
            return _gradient;
        }

        public void Calculate()
        {
        }

        public void GradientPropagation()
        {
            if (TotalGradient != null) AddOperate.Calculate(TotalGradient, _gradient, 1, 0);
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