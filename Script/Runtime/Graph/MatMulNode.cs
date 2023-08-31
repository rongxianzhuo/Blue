using System;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class MatMulNode : IGraphNode
    {

        private readonly IGraphNode _left;
        private readonly IGraphNode _right;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        
        public MatMulNode(IGraphNode left, IGraphNode right)
        {
            _left = left;
            _right = right;
            var size = right.GetOutput().Size / left.GetOutput().Size;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
        }

        public Tensor GetOutput() => _output;

        public Tensor GetGradient() => _gradient;

        public void Forward()
        {
            Op.MatMul(_left.GetOutput()
                , _left.GetOutput().Size
                , _right.GetOutput()
                , _output.Size
                , _output);
        }

        public void Backward()
        {
            Op.MatMul(_right.GetOutput()
                , _output.Size
                , _gradient
                , 1
                , _left.GetGradient());
            Op.MatMul(_left.GetOutput()
                , 1
                , _gradient
                , _output.Size
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