using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ComputationalGraph
    {

        private int _nextAllocateParameterId = 1;

        public int AllocateParameterId()
        {
            return _nextAllocateParameterId++;
        }

        public ComputationalNode ParameterNode(params int[] shape)
        {
            var node = new ComputationalNode(this, true, shape);
            node.AddBackwardOperate(new Operate("Common/GradientIncrease", "CSMain")
                .SetFloat("weight_decay", 0.000f)
                .SetTensor("gradient", node.Gradient)
                .SetTensor("weight", node.Output)
                .SetTensor("total_gradient", node.TotalGradient)
                .SetDispatchSize(node.TotalGradient.FlattenSize));
            return node;
        }

        public ComputationalNode InputNode(params int[] shape)
        {
            return new ComputationalNode(this, false, shape);
        }

        public ComputationalNode Concat(params ComputationalNode[] nodes)
        {
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.Output.Size[1];
            }
            var concat = new ComputationalNode(this, false, nodes[0].Output.Size[0], size);
            concat.AddInputNode(nodes);
            
            var start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t.Output;
                concat.AddForwardOperate(Op.Copy(inputNode, 0, 0
                    , concat.Output, start, size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t.Gradient;
                concat.AddBackwardOperate(Op.Copy(concat.Gradient, start, size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }

            return concat;
        }
        
    }
}