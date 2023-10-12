using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class OperateNode : GraphNode
    {

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public static OperateNode ReLU(GraphNode input)
        {
            return new OperateNode("Graph/ReLU", input.GetOutput().Size[1]
                , new KeyValuePair<string, GraphNode>("input", input));
        }

        public static OperateNode ELU(GraphNode input)
        {
            return new OperateNode("Graph/ELU", input.GetOutput().Size[1]
                , new KeyValuePair<string, GraphNode>("input", input));
        }

        public static OperateNode Sigmoid(GraphNode input)
        {
            return new OperateNode("Graph/Sigmoid", input.GetOutput().Size[1]
                , new KeyValuePair<string, GraphNode>("input", input));
        }

        public OperateNode(string shaderName, int size, params KeyValuePair<string, GraphNode>[] inputs)
        {
            _output = CreateTensor(inputs[0].Value.GetOutput().Size[0], size);
            _gradient = CreateTensor(_output.Size);
            foreach (var pair in inputs)
            {
                InputNodes.Add(pair.Value);
            }
            {
                var op = new Operate(shaderName, "Forward")
                    .SetTensor("rw_output", _output);
                foreach (var pair in inputs)
                {
                    op.SetTensor(pair.Key, pair.Value.GetOutput());
                }
                op.SetDispatchSize(_output.FlattenSize);
                ForwardOperates.Add(op);
            }
            
            foreach (var t in inputs)
            {
                var op = new Operate(shaderName, $"Backward_{t.Key}")
                    .SetTensor("r_output", _output);
                foreach (var pair in inputs)
                {
                    op.SetTensor(pair.Key, pair.Value.GetOutput());
                }
                op.SetTensor("input_gradient", t.Value.GetGradient());
                op.SetTensor("output_gradient", _gradient);
                op.SetDispatchSize(t.Value.GetGradient().FlattenSize);
                BackwardOperates.Add(op);
            }
        }

        public override Tensor GetOutput()
        {
            return _output;
        }

        public override Tensor GetGradient()
        {
            return _gradient;
        }
    }
}