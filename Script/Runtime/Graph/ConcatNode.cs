using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ConcatNode : IGraphNode
    {

        public int ConcatSize { get; private set; }

        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly IGraphNode[] _inputs;
        private readonly List<OperateInstance> _forward = new List<OperateInstance>();
        private readonly List<OperateInstance> _backward = new List<OperateInstance>();

        public ConcatNode(params IGraphNode[] input)
        {
            _inputs = input;
            ConcatSize = 0;
            foreach (var node in _inputs)
            {
                ConcatSize += node.GetOutput().Size[1];
            }

            _output = new Tensor(_inputs[0].GetOutput().Size[0], ConcatSize);
            _gradient = new Tensor(_inputs[0].GetOutput().Size[0], ConcatSize);
            UpdateOperate();
        }

        private void UpdateOperate()
        {
            foreach (var op in _forward)
            {
                op.Destroy();
            }
            foreach (var op in _backward)
            {
                op.Destroy();
            }
            _forward.Clear();
            _backward.Clear();
            var start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetOutput();
                _forward.Add(Op.Copy(inputNode, 0, 0
                    , _output, start, ConcatSize - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetGradient();
                _backward.Add(Op.Copy(_gradient, start, ConcatSize - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
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
            Resize();
            foreach (var op in _forward)
            {
                op.Dispatch();
            }
        }

        public void Backward()
        {
            foreach (var op in _backward)
            {
                op.Dispatch();
            }
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            foreach (var op in _forward)
            {
                op.Destroy();
            }
            foreach (var op in _backward)
            {
                op.Destroy();
            }
            _forward.Clear();
            _backward.Clear();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            foreach (var node in _inputs)
            {
                action(node);
            }
        }

        private void Resize()
        {
            if (_output.Size[0] == _inputs[0].GetOutput().Size[0]) return;
            ConcatSize = 0;
            foreach (var node in _inputs)
            {
                ConcatSize += node.GetOutput().Size[1];
            }
            _output.Resize(_inputs[0].GetOutput().Size[0], ConcatSize);
            _gradient.Resize(_inputs[0].GetOutput().Size[0], ConcatSize);
            UpdateOperate();
        }
    }
}