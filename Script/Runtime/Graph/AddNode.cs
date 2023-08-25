using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    public class AddNode : IGraphNode
    {

        private readonly IGraphNode[] _nodes;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public AddNode(params IGraphNode[] nodes)
        {
            _nodes = nodes;
            var size = nodes[0].GetOutput().count;
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
            CopyOperate.Calculate(_nodes[0].GetOutput(), 0, _output, 0);
            for (var i = 1; i < _nodes.Length; i++)
            {
                AddOperate.Calculate(_output, _nodes[i].GetOutput());
            }
        }

        public void GradientPropagation()
        {
            foreach (var node in _nodes)
            {
                CopyOperate.Calculate(_gradient, 0, node.GetGradient(), 0);
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