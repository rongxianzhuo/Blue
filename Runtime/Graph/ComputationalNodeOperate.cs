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
        
        // 目前仅支持最后两维的转置
        public ComputationalNode Transpose()
        {
            
            var node = new ComputationalNode(new[] { this }, Size);
            (node.Size[^1], node.Size[^2]) = (node.Size[^2], node.Size[^1]);
            node.AddForwardOperate(new Operate("Common/Transpose", "CSMain")
                .SetInt("d1", Size[^1])
                .SetInt("d2", Size[^2])
                .SetTensor("from", this)
                .SetTensor("to", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (Gradient != null) node.AddBackwardOperate(new Operate("Common/Transpose", "CSMain")
                .SetInt("d1", Size[^2])
                .SetInt("d2", Size[^1])
                .SetTensor("from", node.Gradient)
                .SetTensor("to", Gradient)
                .SetDispatchSize(FlattenSize));
            return node;
        }
        
        public ComputationalNode Power(float p)
        {
            var node = new ComputationalNode(new[] { this }, Size);
            node.AddForwardOperate(new Operate("Common/Power", "Forward")
                .SetFloat("p", p)
                .SetTensor("a", this)
                .SetTensor("b", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (Gradient != null) node.AddBackwardOperate(new Operate("Common/Power", "Backward")
                .SetFloat("p", p)
                .SetTensor("a", this)
                .SetTensor("b", node)
                .SetTensor("b_gradient", node.Gradient)
                .SetTensor("a_gradient", Gradient)
                .SetDispatchSize(FlattenSize));
            return node;
        }
        
        public ComputationalNode MaskedFill(Tensor mask, float f)
        {
            var node = new ComputationalNode(new[] { this }, Size);
            node.AddForwardOperate(new Operate("Common/MaskedFill", "Forward")
                .SetFloat("f", f)
                .SetTensor("mask", mask)
                .SetTensor("a", this)
                .SetTensor("b", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (Gradient != null) node.AddBackwardOperate(new Operate("Common/MaskedFill", "Backward")
                .SetFloat("f", f)
                .SetTensor("mask", mask)
                .SetTensor("b_gradient", node.Gradient)
                .SetTensor("a_gradient", Gradient)
                .SetDispatchSize(FlattenSize));
            return node;
        }
        
        public ComputationalNode Softmax(int dim)
        {
            var node = new ComputationalNode(new[] { this }, Size);
            var j2 = 1;
            for (var i = Size.Length - 1; i > dim; i--)
            {
                j2 *= Size[i];
            }
            var j1 = j2 * Size[dim];
            node.AddForwardOperate(new Operate("Common/Softmax", "Forward")
                .SetInt("count", Size[dim])
                .SetInt("j1", j1)
                .SetInt("j2", j2)
                .SetTensor("a", this)
                .SetTensor("b", node)
                .SetDispatchSize(FlattenSize));
            if (Gradient != null) node.AddBackwardOperate(new Operate("Common/Softmax", "Backward")
                .SetInt("count", Size[dim])
                .SetInt("j1", j1)
                .SetInt("j2", j2)
                .SetTensor("b_gradient", node.Gradient)
                .SetTensor("rw_b", node)
                .SetTensor("a_gradient", Gradient)
                .SetDispatchSize(FlattenSize));
            return node;
        }
        
    }
}