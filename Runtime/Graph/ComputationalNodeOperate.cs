using Blue.Core;

namespace Blue.Graph
{
    public partial class ComputationalNode
    {

        public ComputationalNode AdditionAssignment(ComputationalNode other)
        {
            AddInputNode(other);
            AddForwardOperate(new Operate("NN/AdditionAssignment", "Forward")
                .SetInt("other_len", other.FlattenSize)
                .SetTensor("other", other)
                .SetTensor("result", this)
                .SetDispatchSize(FlattenSize));
            if (other.Gradient != null) AddBackwardOperate(new Operate("NN/AdditionAssignment", "Backward")
                .SetInt("other_len", other.FlattenSize)
                .SetInt("result_len", FlattenSize)
                .SetTensor("other_gradient", other.Gradient)
                .SetTensor("result_gradient", Gradient)
                .SetDispatchSize(other.FlattenSize));
            return this;
        }

        public static ComputationalNode operator *(ComputationalNode a, ComputationalNode b)
        {
            var size = a.FlattenSize > b.FlattenSize ? a.Size : b.Size;
            var c = new ComputationalNode(new[] { a, b }, size);
            c.AddForwardOperate(new Operate("NN/Mul", "Forward")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetTensor("a", a)
                .SetTensor("b", b)
                .SetTensor("c", c)
                .SetDispatchSize(c.FlattenSize));
            if (a.Gradient != null) c.AddBackwardOperate(new Operate("NN/Mul", "BackwardA")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetInt("c_len", c.FlattenSize)
                .SetTensor("a_gradient", a.Gradient)
                .SetTensor("b", b)
                .SetTensor("c_gradient", c.Gradient)
                .SetDispatchSize(a.FlattenSize));
            if (b.Gradient != null) c.AddBackwardOperate(new Operate("NN/Mul", "BackwardB")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetInt("c_len", c.FlattenSize)
                .SetTensor("a", a)
                .SetTensor("b_gradient", b.Gradient)
                .SetTensor("c_gradient", c.Gradient)
                .SetDispatchSize(b.FlattenSize));
            return c;
        }
        
        public ComputationalNode MatMul(ComputationalNode other)
        {
            var node = new ComputationalNode(new[] { this, other }, Size[0], other.Size[1]);
            node.AddForwardOperate(new Operate("NN/MatMul", "Forward")
                .SetInt("wl", Size[1])
                .SetInt("wr", other.Size[1])
                .SetTensor("left", this)
                .SetTensor("right", other)
                .SetTensor("result", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (Gradient != null) node.AddBackwardOperate(new Operate("NN/MatMul", "BackwardLeft")
                .SetInt("wl", Size[1])
                .SetInt("wr", other.Size[1])
                .SetTensor("right", other)
                .SetTensor("result_gradient", node.Gradient)
                .SetTensor("left_gradient", Gradient)
                .SetDispatchSize(FlattenSize));
            
            if (other.Gradient != null) node.AddBackwardOperate(new Operate("NN/MatMul", "BackwardRight")
                .SetInt("hl", Size[0])
                .SetInt("wl", Size[1])
                .SetInt("wr", other.Size[1])
                .SetTensor("left", this)
                .SetTensor("result_gradient", node.Gradient)
                .SetTensor("right_gradient", other.Gradient)
                .SetDispatchSize(other.FlattenSize));
            return node;
        }
        
    }
}