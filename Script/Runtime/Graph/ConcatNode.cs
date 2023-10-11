using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ConcatNode : BasicGraphNode
    {

        private readonly int _size;
        private readonly GraphNode[] _inputs;

        public ConcatNode(params GraphNode[] input)
        {
            _inputs = input;
            _size = 0;
            foreach (var node in _inputs)
            {
                _size += node.GetOutput().Size[1];
            }
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward)
        {
            var start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetOutput();
                forward.Add(Op.Copy(inputNode, 0, 0
                    , GetOutput(), start, _size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in _inputs)
            {
                var inputNode = t.GetGradient();
                backward.Add(Op.Copy(GetGradient(), start, _size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
        }

        protected override void GetOutputSize(out int batchSize, out int size)
        {
            batchSize = _inputs[0].GetOutput().Size[0];
            size = _size;
        }

        protected override void OnDestroy()
        {
        }

        public override void ForeachInputNode(Action<GraphNode> action)
        {
            foreach (var node in _inputs)
            {
                action(node);
            }
        }
    }
}