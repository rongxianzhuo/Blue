using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ConcatNode : GraphNode
    {

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public ConcatNode(params GraphNode[] input)
        {
            InputNodes.AddRange(input);
            var size = 0;
            foreach (var node in input)
            {
                size += node.GetOutput().Size[1];
            }
            _output = new Tensor(input[0].GetOutput().Size[0], size);
            _gradient = new Tensor(_output.Size);
            
            var start = 0;
            foreach (var t in InputNodes)
            {
                var inputNode = t.GetOutput();
                ForwardOperates.Add(Op.Copy(inputNode, 0, 0
                    , _output, start, size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in InputNodes)
            {
                var inputNode = t.GetGradient();
                BackwardOperates.Add(Op.Copy(_gradient, start, size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
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

        protected override void OnDestroy()
        {
            _output.Release();
            _gradient.Release();
        }
    }
}