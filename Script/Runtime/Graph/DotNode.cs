using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    public class DotNode : IGraphNode
    {

        private readonly IGraphNode _left;
        private readonly IGraphNode _right;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;
        
        public DotNode(IGraphNode left, IGraphNode right)
        {
            _left = left;
            _right = right;
            var size = right.GetOutput().count / left.GetOutput().count;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }

        public ComputeBuffer GetOutput() => _output;

        public ComputeBuffer GetGradient() => _gradient;

        public void Calculate()
        {
            DotOperate.Calculate(_left.GetOutput().count, 1, _left.GetOutput(), _right.GetOutput(), _output);
        }

        public void GradientPropagation()
        {
            DotOperate.Calculate(1, _left.GetOutput().count, _gradient, _right.GetOutput(), _left.GetGradient());
            MatMulOperate.Calculate(1, _gradient, _left.GetOutput(), _right.GetGradient());
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_left);
            action(_right);
        }
    }
}