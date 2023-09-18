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
        
        private static Operate _copyOp;
        private static Operate GetCopyOp() => _copyOp ??= new Operate("Common/Copy", "CSMain"
            , "src_start", "dst_start", "src_interval", "dst_interval", "stride", "src_buffer", "dst_buffer");
        public static void Copy(Tensor src, int srcStart, int srcInterval, Tensor dst, int dstStart, int dstInterval, int stride, int length)
        {
            GetCopyOp().CreateTask()
                .SetInt(srcStart)
                .SetInt(dstStart)
                .SetInt(srcInterval)
                .SetInt(dstInterval)
                .SetInt(stride)
                .SetTensor(src)
                .SetTensor(dst)
                .Dispatch(length);
        }
        
        private static Operate _clearOp;
        private static Operate GetClearOp() => _clearOp ??= new Operate("Common/Clear", "CSMain"
            , "clear_value", "buffer");
        public static void Clear(Tensor buffer, float clearValue)
        {
            GetClearOp().CreateTask()
                .SetFloat(clearValue)
                .SetTensor(buffer)
                .Dispatch(buffer.FlattenSize);
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
        
        private static Operate _crossEntropyLossOp;
        private static Operate GetCrossEntropyLossOp() => _crossEntropyLossOp ??= new Operate($"LossFunction/CrossEntropyLoss", "CSMain"
            , "total_count", "output", "target", "gradient");
        public static void CrossEntropyLoss(Tensor output, Tensor target, Tensor gradient)
        {
            GetCrossEntropyLossOp().CreateTask()
                .SetInt(target.Size[1])
                .SetTensor(output)
                .SetTensor(target)
                .SetTensor(gradient)
                .Dispatch(target.FlattenSize);
        }
        
        private static Operate _l2LossOp;
        private static Operate GetL2LossOp() => _l2LossOp ??= new Operate($"LossFunction/L2Loss", "CSMain"
            , "output", "target", "gradient");
        public static void L2Loss(Tensor output, Tensor target, Tensor gradient)
        {
            GetL2LossOp().CreateTask()
                .SetTensor(output)
                .SetTensor(target)
                .SetTensor(gradient)
                .Dispatch(target.FlattenSize);
        }
        
    }
}