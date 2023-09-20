using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class OperateNode : BasicGraphNode
    {

        private readonly int _size;
        private readonly string _shaderName;
        private readonly KeyValuePair<string, IGraphNode>[] _inputs;

        public static OperateNode ReLU(IGraphNode input)
        {
            return new OperateNode("Graph/ReLU", input.GetOutput().Size[1]
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode ELU(IGraphNode input)
        {
            return new OperateNode("Graph/ELU", input.GetOutput().Size[1]
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public static OperateNode Sigmoid(IGraphNode input)
        {
            return new OperateNode("Graph/Sigmoid", input.GetOutput().Size[1]
                , new KeyValuePair<string, IGraphNode>("input", input));
        }

        public OperateNode(string shaderName, int size, params KeyValuePair<string, IGraphNode>[] inputs)
        {
            _size = size;
            _shaderName = shaderName;
            _inputs = inputs;
        }

        protected override void UpdateOperate(int batchSize, List<OperateInstance> forward, List<OperateInstance> backward)
        {
            {
                var op = new OperateInstance(_shaderName, "Forward")
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
                var op = new OperateInstance(_shaderName, $"Backward_{t.Key}")
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

        public override void ForeachInputNode(Action<IGraphNode> action)
        {
            foreach (var node in _inputs)
            {
                action(node.Value);
            }
        }
    }
}