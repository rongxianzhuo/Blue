using Blue.Graph;
using Blue.Kit;

namespace Blue.Runtime.NN
{
    public class Concat : Module
    {
        public override ComputationalNode Build(params ComputationalNode[] nodes)
        {
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.Size[1];
            }
            var concat = new ComputationalNode(nodes, nodes[0].Size[0], size);
            
            var start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t;
                concat.AddForwardOperate(Op.Copy(inputNode, 0, 0
                    , concat, start, size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t.Gradient;
                if (inputNode != null) concat.AddBackwardOperate(Op.Copy(concat.Gradient, start, size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += t.Size[1];
            }

            return concat;
        }
    }
}