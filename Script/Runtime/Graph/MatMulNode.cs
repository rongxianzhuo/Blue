using System;
using UnityEngine;
using Blue.Kit;

namespace Blue.Graph
{
    public class MatMulNode : IGraphNode
    {

        private readonly IGraphNode _left;
        private readonly IGraphNode _right;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;
        
        public MatMulNode(IGraphNode left, IGraphNode right)
        {
            _left = left;
            _right = right;
            var size = right.GetOutput().count / left.GetOutput().count;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }

        public ComputeBuffer GetOutput() => _output;

        public ComputeBuffer GetGradient() => _gradient;

        public void Forward()
        {
            Op.MatMul(_left.GetOutput()
                , _left.GetOutput().count
                , _right.GetOutput()
                , _output.count
                , _output);
        }

        public void Backward()
        {
            Op.MatMul(_right.GetOutput()
                , _output.count
                , _gradient
                , 1
                , _left.GetGradient());
            Op.MatMul(_left.GetOutput()
                , 1
                , _gradient
                , _output.count
                , _right.GetGradient());
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