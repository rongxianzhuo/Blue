using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ConcatNode : BasicGraphNode
    {

        private readonly int _size;

        public ConcatNode(params GraphNode[] input)
        {
            InputNodes.AddRange(input);
            _size = 0;
            foreach (var node in input)
            {
                _size += node.GetOutput().Size[1];
            }
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward)
        {
            var start = 0;
            foreach (var t in InputNodes)
            {
                var inputNode = t.GetOutput();
                forward.Add(Op.Copy(inputNode, 0, 0
                    , GetOutput(), start, _size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in InputNodes)
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
            batchSize = InputNodes[0].GetOutput().Size[0];
            size = _size;
        }

        protected override void OnDestroy()
        {
        }
    }
}