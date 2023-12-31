using Blue.Core;

namespace Blue.Kit
{
    public static class Op
    {
        
        public static Operate Translate(Tensor buffer, float weight, float bias)
        {
            return new Operate("Common/Translate", "CSMain")
                .SetFloat("weight", weight)
                .SetFloat("bias", bias)
                .SetTensor("rw_buffer1", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static Operate MatMul(Tensor left, Tensor right, Tensor result)
        {
            return new Operate("Common/MatMul", "CSMain")
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("left", left)
                .SetTensor("right", right)
                .SetTensor("result", result)
                .SetDispatchSize(result.FlattenSize);
        }
        
        public static Operate Increment(Tensor buffer, Tensor other)
        {
            return new Operate("Common/Increment", "CSMain")
                .SetInt("other_count", other.FlattenSize)
                .SetTensor("r_buffer1", other)
                .SetTensor("rw_buffer1", buffer)
                .SetDispatchSize(buffer.FlattenSize);
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
        
        public static Operate L2Loss(Tensor output, Tensor target, Tensor gradient)
        {
            return new Operate("LossFunction/L2Loss", "CSMain")
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
        
    }
}