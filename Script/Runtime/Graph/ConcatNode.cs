using System;
using Blue.Kit;
using UnityEngine;

namespace Blue.Graph
{
    public class ConcatNode : IGraphNode
    {

        private readonly IGraphNode[] _nodes;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public ConcatNode(params IGraphNode[] nodes)
        {
            _nodes = nodes;
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.GetOutput().count;
            }
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

        public void Forward()
        {
            var i = 0;
            foreach (var node in _nodes)
            {
                Op.Copy(node.GetOutput(), 0, _output, i, node.GetOutput().count);
                i += node.GetOutput().count;
            }
        }

        public void Backward()
        {
            var i = 0;
            foreach (var node in _nodes)
            {
                Op.Copy(_output, i, node.GetGradient(), 0, node.GetGradient().count);
                i += node.GetOutput().count;
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