using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class OperateNode : IGraphNode
    {
        
        private readonly Operate _forward;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly List<IGraphNode> _inputs = new List<IGraphNode>();
        private readonly List<Operate> _backward = new List<Operate>();

        public static OperateNode ReLU(IGraphNode input)
        {
            return new OperateNode("Graph/ReLU", input.GetOutput().FlattenSize
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode ELU(IGraphNode input)
        {
            return new OperateNode("Graph/ELU", input.GetOutput().FlattenSize
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode Sigmoid(IGraphNode input)
        {
            return new OperateNode("Graph/Sigmoid", input.GetOutput().FlattenSize
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public OperateNode(string shaderName, int size, params KeyValuePair<string, IGraphNode>[] inputs)
        {
            var properties = new List<string>(inputs.Length + 2);
            properties.Add("rw_output");
            foreach (var pair in inputs)
            {
                properties.Add(pair.Key);
                _inputs.Add(pair.Value);
            }
            _forward = new Operate(shaderName, "Forward", properties.ToArray());
            properties[0] = "r_output";
            properties.Add("input_gradient");
            properties.Add("output_gradient");
            var propertiesArray = properties.ToArray();
            for (var i = 0; i < inputs.Length; i++)
            {
                _backward.Add(new Operate(shaderName, $"Backward_{inputs[i].Key}", propertiesArray));
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
            var handler = _forward.CreateTask();
            handler.SetTensor(_output);
            foreach (var node in _inputs)
            {
                handler.SetTensor(node.GetOutput());
            }
            handler.Dispatch(_output.FlattenSize);
        }

        public void Backward()
        {
            for (var i = 0; i < _inputs.Count; i++)
            {
                var op = _backward[i];
                var handler = op.CreateTask();
                handler.SetTensor(_output);
                foreach (var node in _inputs)
                {
                    handler.SetTensor(node.GetOutput());
                }
                handler.SetTensor(_inputs[i].GetGradient());
                handler.SetTensor(_gradient);
                handler.Dispatch(_inputs[i].GetOutput().FlattenSize);
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
    }
}