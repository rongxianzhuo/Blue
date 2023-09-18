using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class OperateNode : IGraphNode
    {

        private readonly string _shaderName;
        private OperateInstance _forward;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly KeyValuePair<string, IGraphNode>[] _inputs;
        private readonly List<OperateInstance> _backward = new List<OperateInstance>();

        public static OperateNode ReLU(IGraphNode input)
        {
            return new OperateNode("Graph/ReLU", input.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode ELU(IGraphNode input)
        {
            return new OperateNode("Graph/ELU", input.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode Sigmoid(IGraphNode input)
        {
            return new OperateNode("Graph/Sigmoid", input.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public OperateNode(string shaderName, int[] size, params KeyValuePair<string, IGraphNode>[] inputs)
        {
            _shaderName = shaderName;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
            _inputs = inputs;
            UpdateOperate();
        }

        private void UpdateOperate()
        {
            _forward?.Destroy();
            _forward = new OperateInstance(_shaderName, "Forward")
                .SetTensor("rw_output", _output);
            foreach (var pair in _inputs)
            {
                _forward.SetTensor(pair.Key, pair.Value.GetOutput());
            }
            _forward.SetDispatchSize(_output.FlattenSize);
            
            foreach (var t in _inputs)
            {
                var op = new OperateInstance(_shaderName, $"Backward_{t.Key}")
                    .SetTensor("r_output", _output);
                foreach (var pair in _inputs)
                {
                    op.SetTensor(pair.Key, pair.Value.GetOutput());
                }
                op.SetTensor("input_gradient", t.Value.GetGradient());
                op.SetTensor("output_gradient", _gradient);
                op.SetDispatchSize(t.Value.GetGradient().FlattenSize);
                _backward.Add(op);
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
            var resize = _output.Resize(_inputs[0].Value.GetOutput().Size);
            _gradient.Resize(_inputs[0].Value.GetOutput().Size);
            if (resize) UpdateOperate();
            _forward.Dispatch();
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
            _forward?.Destroy();
            _forward = null;
            foreach (var op in _backward)
            {
                op.Destroy();
            }
            _backward.Clear();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            foreach (var node in _inputs)
            {
                action(node.Value);
            }
        }
    }
}