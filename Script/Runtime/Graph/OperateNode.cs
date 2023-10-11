using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class OperateNode : BasicGraphNode
    {

        private readonly int _size;
        private readonly string _shaderName;
        private readonly KeyValuePair<string, GraphNode>[] _inputs;

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
            _size = size;
            _shaderName = shaderName;
            _inputs = inputs;
            foreach (var pair in inputs)
            {
                InputNodes.Add(pair.Value);
            }
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward)
        {
            {
                var op = new Operate(_shaderName, "Forward")
                    .SetTensor("rw_output", GetOutput());
                foreach (var pair in _inputs)
                {
                    op.SetTensor(pair.Key, pair.Value.GetOutput());
                }
                op.SetDispatchSize(GetOutput().FlattenSize);
                forward.Add(op);
            }
            
            foreach (var t in _inputs)
            {
                var op = new Operate(_shaderName, $"Backward_{t.Key}")
                    .SetTensor("r_output", GetOutput());
                foreach (var pair in _inputs)
                {
                    op.SetTensor(pair.Key, pair.Value.GetOutput());
                }
                op.SetTensor("input_gradient", t.Value.GetGradient());
                op.SetTensor("output_gradient", GetGradient());
                op.SetDispatchSize(t.Value.GetGradient().FlattenSize);
                backward.Add(op);
            }
        }

        protected override void GetOutputSize(out int batchSize, out int size)
        {
            batchSize = _inputs[0].Value.GetOutput().Size[0];
            size = _size;
        }

        protected override void OnDestroy()
        {
        }
    }
}