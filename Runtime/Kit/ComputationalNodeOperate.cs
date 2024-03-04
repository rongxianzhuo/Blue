using Blue.Core;
using Blue.Graph;

namespace Blue.Kit
{
    public static class ComputationalNodeOperate
    {

        public static ComputationalNode Add(this ComputationalNode node, ComputationalNode other)
        {
            node.AddForwardOperate(new Operate("Common/Add", "Forward")
                .SetInt("other_len", other.FlattenSize)
                .SetTensor("other", other)
                .SetTensor("result", node)
                .SetDispatchSize(node.FlattenSize));
            if (other.Gradient != null) node.AddBackwardOperate(new Operate("Common/Add", "Backward")
                .SetInt("other_len", other.FlattenSize)
                .SetInt("result_len", node.FlattenSize)
                .SetTensor("other_gradient", other.Gradient)
                .SetTensor("result_gradient", node.Gradient)
                .SetDispatchSize(other.FlattenSize));
            return node;
        }
        
        public static ComputationalNode MatMul(this ComputationalNode node, ComputationalNode left, ComputationalNode right)
        {
            node.AddForwardOperate(new Operate("Common/MatMul", "Forward")
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("left", left)
                .SetTensor("right", right)
                .SetTensor("result", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (left.Gradient != null) node.AddBackwardOperate(new Operate("Common/MatMul", "BackwardLeft")
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("right", right)
                .SetTensor("result_gradient", node.Gradient)
                .SetTensor("left_gradient", left.Gradient)
                .SetDispatchSize(left.FlattenSize));
            
            if (right.Gradient != null) node.AddBackwardOperate(new Operate("Common/MatMul", "BackwardRight")
                .SetInt("hl", left.Size[0])
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("left", left)
                .SetTensor("result_gradient", node.Gradient)
                .SetTensor("right_gradient", right.Gradient)
                .SetDispatchSize(right.FlattenSize));
            return node;
        }
        
    }
}