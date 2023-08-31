using System;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ConcatNode : IGraphNode
    {

        private readonly IGraphNode[] _nodes;
        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public ConcatNode(params IGraphNode[] nodes)
        {
            _nodes = nodes;
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.GetOutput().Size;
            }
            _output = new Tensor(size);
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
            var i = 0;
            foreach (var node in _nodes)
            {
                Op.Copy(node.GetOutput(), 0, _output, i, node.GetOutput().Size);
                i += node.GetOutput().Size;
            }
        }

        public void Backward()
        {
            var i = 0;
            foreach (var node in _nodes)
            {
                Op.Copy(_output, i, node.GetGradient(), 0, node.GetGradient().Size);
                i += node.GetOutput().Size;
            }
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            foreach (var node in _nodes)
            {
                action(node);
            }
        }
    }
}