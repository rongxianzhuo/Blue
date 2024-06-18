using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public partial class ComputationalNode
    {

        public ComputationalNode AddInPlace(ComputationalNode other)
        {
            AddInputNode(other);
            AddForwardOperate(new Operate("NN/AddInPlace", "Forward")
                .SetInt("other_len", other.FlattenSize)
                .SetTensor("other", other)
                .SetTensor("result", this)
                .SetDispatchSize(FlattenSize));
            if (other.Gradient != null) AddBackwardOperate(new Operate("NN/AddInPlace", "Backward")
                .SetInt("other_len", other.FlattenSize)
                .SetInt("result_len", FlattenSize)
                .SetTensor("other_gradient", other.Gradient)
                .SetTensor("result_gradient", Gradient)
                .SetDispatchSize(other.FlattenSize));
            return this;
        }

        public static ComputationalNode operator +(ComputationalNode a, ComputationalNode b)
        {
            var size = a.FlattenSize > b.FlattenSize ? a.Size : b.Size;
            var c = new ComputationalNode(new[] { a, b }, size);
            c.AddForwardOperate(new Operate("O2", "ForwardAdd")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input1", a)
                .SetTensor("input2", b)
                .SetTensor("output", c)
                .SetDispatchSize(c.FlattenSize));
            var aStrideOrder = a.CalculateStrideOrder();
            if (a.Gradient != null) c.AddBackwardOperate(new Operate("O2", "BackwardAdd1")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input1_gradient", a.Gradient, aStrideOrder)
                .SetTensor("output_gradient", c.Gradient, aStrideOrder)
                .SetDispatchSize(a.FlattenSize));
            var bStrideOrder = b.CalculateStrideOrder();
            if (b.Gradient != null) c.AddBackwardOperate(new Operate("O2", "BackwardAdd2")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input2_gradient", b.Gradient, bStrideOrder)
                .SetTensor("output_gradient", c.Gradient, bStrideOrder)
                .SetDispatchSize(b.FlattenSize));
            return c;
        }

        public static ComputationalNode operator *(ComputationalNode a, ComputationalNode b)
        {
            var size = a.FlattenSize > b.FlattenSize ? a.Size : b.Size;
            var c = new ComputationalNode(new[] { a, b }, size);
            c.AddForwardOperate(new Operate("O2", "ForwardMul")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input1", a)
                .SetTensor("input2", b)
                .SetTensor("output", c)
                .SetDispatchSize(c.FlattenSize));
            var aStrideOrder = a.CalculateStrideOrder();
            if (a.Gradient != null) c.AddBackwardOperate(new Operate("O2", "BackwardMul1")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input1_gradient", a.Gradient, aStrideOrder)
                .SetTensor("input2", b, aStrideOrder)
                .SetTensor("output_gradient", c.Gradient, aStrideOrder)
                .SetDispatchSize(a.FlattenSize));
            var bStrideOrder = b.CalculateStrideOrder();
            if (b.Gradient != null) c.AddBackwardOperate(new Operate("O2", "BackwardMul2")
                .SetInt("dim", a.Size.Length)
                .SetTensor("input2_gradient", b.Gradient, bStrideOrder)
                .SetTensor("input1", a, bStrideOrder)
                .SetTensor("output_gradient", c.Gradient, bStrideOrder)
                .SetDispatchSize(b.FlattenSize));
            return c;
        }
        
        public ComputationalNode MatMul(ComputationalNode other)
        {
            var node = new ComputationalNode(new[] { this, other }, Size[0], other.Size[1]);
            node.AddForwardOperate(new Operate("NN/MatMul", "Forward")
                .SetInt("lw", other.Size[0])
                .SetTensor("left", this)
                .SetTensor("right", other)
                .SetTensor("result", node)
                .SetDispatchSize(node.FlattenSize));
            node.AddBackwardOperate(new Operate("NN/MatMul", "BackwardLeft")
                .SetInt("lw", Size[1])
                .SetInt("lh", Size[0])
                .SetTensor("left_gradient", Gradient)
                .SetTensor("right", other)
                .SetTensor("result_gradient", node.Gradient)
                .SetDispatchSize(FlattenSize));
            node.AddBackwardOperate(new Operate("NN/MatMul", "BackwardRight")
                .SetInt("t", FlattenSize / other.FlattenSize)
                .SetInt("lh", Size[0])
                .SetInt("lw", Size[1])
                .SetInt("rw", other.Size[1])
                .SetTensor("left", this)
                .SetTensor("right_gradient", other.Gradient)
                .SetTensor("result_gradient", node.Gradient)
                .SetDispatchSize(other.FlattenSize));
            return node;
        }
        
        public ComputationalNode Transpose(int dim1, int dim2)
        {
            var shape = new List<int>(Size);
            var stride = new List<int>(StrideInMemory);
            (shape[dim1], shape[dim2]) = (shape[dim2], shape[dim1]);
            (stride[dim1], stride[dim2]) = (stride[dim2], stride[dim1]);
            var node = new ComputationalNode(shape, this, stride);
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
            return null;
        }

        public ComputationalNode ReLU()
        {
            var node = new ComputationalNode(new[] { this }, Size);
            const string shaderName = "Activation";
            node.AddForwardOperate(new Operate(shaderName, "ForwardReLU")
                .SetInt("dim", Size.Length)
                .SetTensor("rw_output", node)
                .SetTensor("input", this)
                .SetDispatchSize(FlattenSize));
            
            var inputStrideOrder = Gradient.CalculateStrideOrder();
            if (Gradient != null) node.AddBackwardOperate(new Operate(shaderName, "BackwardReLU")
                .SetInt("dim", Size.Length)
                .SetTensor("r_output", node, inputStrideOrder)
                .SetTensor("input_gradient", Gradient, inputStrideOrder)
                .SetTensor("output_gradient", node.Gradient, inputStrideOrder)
                .SetDispatchSize(Gradient.FlattenSize));

            return node;
        }

        public ComputationalNode ELU()
        {
            var node = new ComputationalNode(new[] { this }, Size);
            const string shaderName = "Activation";
            node.AddForwardOperate(new Operate(shaderName, "ForwardELU")
                .SetInt("dim", Size.Length)
                .SetTensor("rw_output", node)
                .SetTensor("input", this)
                .SetDispatchSize(FlattenSize));
            
            var inputStrideOrder = Gradient.CalculateStrideOrder();
            if (Gradient != null) node.AddBackwardOperate(new Operate(shaderName, "BackwardELU")
                .SetInt("dim", Size.Length)
                .SetTensor("r_output", node, inputStrideOrder)
                .SetTensor("input_gradient", Gradient, inputStrideOrder)
                .SetTensor("output_gradient", node.Gradient, inputStrideOrder)
                .SetDispatchSize(Gradient.FlattenSize));

            return node;
        }

        public ComputationalNode Sigmoid()
        {
            var node = new ComputationalNode(new[] { this }, Size);
            const string shaderName = "Activation";
            node.AddForwardOperate(new Operate(shaderName, "ForwardSigmoid")
                .SetInt("dim", Size.Length)
                .SetTensor("rw_output", node)
                .SetTensor("input", this)
                .SetDispatchSize(FlattenSize));
            
            var inputStrideOrder = Gradient.CalculateStrideOrder();
            if (Gradient != null) node.AddBackwardOperate(new Operate(shaderName, "BackwardSigmoid")
                .SetInt("dim", Size.Length)
                .SetTensor("r_output", node, inputStrideOrder)
                .SetTensor("input_gradient", Gradient, inputStrideOrder)
                .SetTensor("output_gradient", node.Gradient, inputStrideOrder)
                .SetDispatchSize(Gradient.FlattenSize));

            return node;
        }

        public ComputationalNode Tanh()
        {
            var node = new ComputationalNode(new[] { this }, Size);
            const string shaderName = "Activation";
            node.AddForwardOperate(new Operate(shaderName, "ForwardTanh")
                .SetInt("dim", Size.Length)
                .SetTensor("rw_output", node)
                .SetTensor("input", this)
                .SetDispatchSize(FlattenSize));

            var inputStrideOrder = Gradient.CalculateStrideOrder();
            if (Gradient != null) node.AddBackwardOperate(new Operate(shaderName, "BackwardTanh")
                .SetInt("dim", Size.Length)
                .SetTensor("r_output", node, inputStrideOrder)
                .SetTensor("input_gradient", Gradient, inputStrideOrder)
                .SetTensor("output_gradient", node.Gradient, inputStrideOrder)
                .SetDispatchSize(Gradient.FlattenSize));

            return node;
        }

        public ComputationalNode Dropout(float dropout)
        {
            var weightArray = new float[FlattenSize];
            var node = new ComputationalNode(false, Size);
            node.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= dropout ? 1f : 0f;
                }
                node.SetData(weightArray);
            }));
            return this * node;
        }
        
    }
}