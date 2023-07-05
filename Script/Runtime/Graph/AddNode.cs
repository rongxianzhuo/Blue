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
            SetOperate.Calculate(_output, 0);
            foreach (var node in _nodes)
            {
                AddOperate.Calculate(_output, node.GetOutput(), 1, 0);
            }
        }

        public void GradientPropagation()
        {
            foreach (var node in _nodes)
            {
                SetOperate.Calculate(node.GetGradient(), 0);
                AddOperate.Calculate(node.GetGradient(), _gradient, 1, 0);
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