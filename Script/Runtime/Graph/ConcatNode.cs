using System;
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
            var start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetOutput();
                Op.Copy(inputNode, 0, 0
                    , _output, start, ConcatSize - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize);
                start += inputNode.Size[1];
            }
        }

        public void Backward()
        {
            var start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetGradient();
                Op.Copy(_gradient, start, ConcatSize - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize);
                start += inputNode.Size[1];
            }
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
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
        }
    }
}