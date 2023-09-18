using Blue.Core;

namespace Blue.Kit
{
    public static class Op
    {
        
        public static OperateInstance Translate(Tensor buffer, float weight, float bias)
        {
            return new OperateInstance("Common/Translate", "CSMain")
                .SetFloat("weight", weight)
                .SetFloat("bias", bias)
                .SetTensor("rw_buffer1", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static OperateInstance MatMul(Tensor left, Tensor right, Tensor result)
        {
            return new OperateInstance("Common/MatMul", "CSMain")
                .SetInt("wl", left.Size[1])
                .SetInt("wr", right.Size[1])
                .SetTensor("left", left)
                .SetTensor("right", right)
                .SetTensor("result", result)
                .SetDispatchSize(result.FlattenSize);
        }
        
        public static OperateInstance Increment(Tensor buffer, Tensor other)
        {
            return new OperateInstance("Common/Increment", "CSMain")
                .SetInt("other_count", other.FlattenSize)
                .SetTensor("r_buffer1", other)
                .SetTensor("rw_buffer1", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static OperateInstance Copy(Tensor src, int srcStart, int srcInterval, Tensor dst, int dstStart, int dstInterval, int stride, int length)
        {
            return new OperateInstance("Common/Copy", "CSMain")
                .SetInt("src_start", srcStart)
                .SetInt("dst_start", dstStart)
                .SetInt("src_interval", srcInterval)
                .SetInt("dst_interval", dstInterval)
                .SetInt("stride", stride)
                .SetTensor("src_buffer", src)
                .SetTensor("dst_buffer", dst)
                .SetDispatchSize(length);
        }
        
        public static OperateInstance Clear(Tensor buffer, float clearValue)
        {
            return new OperateInstance("Common/Clear", "CSMain")
                .SetFloat("clear_value", clearValue)
                .SetTensor("buffer", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static OperateInstance Transpose(Tensor src, Tensor dst)
        {
            return new OperateInstance("Common/Transpose", "CSMain")
                .SetInt("src_height", src.Size[0])
                .SetInt("src_width", src.Size[1])
                .SetTensor("from", src)
                .SetTensor("to", dst)
                .SetDispatchSize(dst.FlattenSize);
        }
        
        public static OperateInstance CrossEntropyLoss(Tensor output, Tensor target, Tensor gradient)
        {
            return new OperateInstance("LossFunction/CrossEntropyLoss", "CSMain")
                .SetInt("total_count", target.Size[1])
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetDispatchSize(target.FlattenSize);
        }
        
        public static OperateInstance L2Loss(Tensor output, Tensor target, Tensor gradient)
        {
            return new OperateInstance("LossFunction/L2Loss", "CSMain")
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetDispatchSize(target.FlattenSize);
        }
        
    }
}