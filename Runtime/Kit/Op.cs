using Blue.Core;
using Blue.Graph;

namespace Blue.Kit
{
    public static class Op
    {
        
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
                .SetInt("batch_size", left.Size[0])
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("left", left)
                .SetTensor("result_gradient", node.Gradient)
                .SetTensor("right_gradient", right.Gradient)
                .SetDispatchSize(right.FlattenSize));
            return node;
        }
        
        public static Operate Copy(Tensor src, int srcStart, int srcInterval, Tensor dst, int dstStart, int dstInterval, int stride, int length)
        {
            return new Operate("Common/Copy", "CSMain")
                .SetInt("src_start", srcStart)
                .SetInt("dst_start", dstStart)
                .SetInt("src_interval", srcInterval)
                .SetInt("dst_interval", dstInterval)
                .SetInt("stride", stride)
                .SetTensor("src_buffer", src)
                .SetTensor("dst_buffer", dst)
                .SetDispatchSize(length);
        }
        
        public static Operate Clear(Tensor buffer, float clearValue)
        {
            return new Operate("Common/Clear", "CSMain")
                .SetFloat("clear_value", clearValue)
                .SetTensor("buffer", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static Operate Transpose(Tensor src, Tensor dst)
        {
            return new Operate("Common/Transpose", "CSMain")
                .SetInt("src_height", src.Size[0])
                .SetInt("src_width", src.Size[1])
                .SetTensor("from", src)
                .SetTensor("to", dst)
                .SetDispatchSize(dst.FlattenSize);
        }
        
        public static Operate CrossEntropyLoss(Tensor output, Tensor target, Tensor gradient)
        {
            return new Operate("LossFunction/CrossEntropyLoss", "CSMain")
                .SetInt("total_count", target.Size[1])
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetDispatchSize(target.FlattenSize);
        }

        public static ComputationalNode Add(this ComputationalNode node, ComputationalNode other)
        {
            node.AddForwardOperate(new Operate("Common/Add", "Forward")
                .SetInt("other_len", other.FlattenSize)
                .SetTensor("other", other)
                .SetTensor("result", node)
                .SetDispatchSize(node.FlattenSize));
            if (other.Gradient != null) node.AddBackwardOperate(new Operate("Common/Add", "Backward")
                .SetInt("batch_size", node.FlattenSize / other.FlattenSize)
                .SetInt("other_len", other.FlattenSize)
                .SetInt("result_len", node.FlattenSize)
                .SetTensor("other_gradient", other.Gradient)
                .SetTensor("result_gradient", node.Gradient)
                .SetDispatchSize(other.FlattenSize));
            return node;
        }
        
        public static Operate L2Loss(Tensor output, Tensor target, Tensor gradient)
        {
            return new Operate("LossFunction/L2Loss", "CSMain")
                .SetInt("n", output.Size[output.Size.Length - 1])
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetDispatchSize(target.FlattenSize);
        }
        
        public static Operate WeightedL1Loss(Tensor output, Tensor target, Tensor gradient, Tensor weight)
        {
            return new Operate("LossFunction/WeightedL1Loss", "CSMain")
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetTensor("weight", weight)
                .SetDispatchSize(target.FlattenSize);
        }
        
        public static Operate Variance(Tensor input, Tensor result)
        {
            return new Operate("Common/Variance", "CSMain")
                .SetInt("n", input.Size[1])
                .SetTensor("buffer", input)
                .SetTensor("result", result)
                .SetDispatchSize(input.Size[0]);
        }
        
    }
}