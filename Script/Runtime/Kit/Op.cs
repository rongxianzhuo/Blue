using Blue.Core;

namespace Blue.Kit
{
    public static class Op
    {
        
        private static Operate _translateOp;
        private static Operate GetTranslateOp() => _translateOp ??= new Operate("Common/Translate", "CSMain"
            , "weight", "bias", "rw_buffer1");
        public static void Translate(Tensor buffer, float weight, float bias)
        {
            GetTranslateOp().CreateTask()
                .SetFloat(weight)
                .SetFloat(bias)
                .SetTensor(buffer)
                .Dispatch(buffer.FlattenSize);
        }

        private static Operate _matMulOp;
        private static Operate GetMatMulOp() => _matMulOp ??= new Operate("Common/MatMul", "CSMain"
            , "wl"
            , "wr"
            , "left"
            , "right"
            , "result");
        public static void MatMul(Tensor left, Tensor right, Tensor result)
        {
            GetMatMulOp().CreateTask()
                .SetInt(left.Size[1])
                .SetInt(right.Size[1])
                .SetTensor(left)
                .SetTensor(right)
                .SetTensor(result)
                .Dispatch(result.FlattenSize);
        }
        
        private static Operate _incrementOp;
        private static Operate GetIncrementOp() => _incrementOp ??= new Operate("Common/Increment", "CSMain"
            , "other_count", "r_buffer1", "rw_buffer1");
        public static void Increment(Tensor buffer, Tensor other)
        {
            GetIncrementOp().CreateTask()
                .SetInt(other.FlattenSize)
                .SetTensor(other)
                .SetTensor(buffer)
                .Dispatch(buffer.FlattenSize);
        }
        
        private static Operate _copyOp;
        private static Operate GetCopyOp() => _copyOp ??= new Operate("Common/Copy", "CSMain"
            , "r_buffer1", "src_offset", "rw_buffer1", "dst_offset");
        public static void Copy(Tensor src, int srcStartIndex, Tensor dst, int dstStartIndex, int length)
        {
            GetCopyOp().CreateTask()
                .SetTensor(src)
                .SetInt(srcStartIndex)
                .SetTensor(dst)
                .SetInt(dstStartIndex)
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
        
        private static Operate _transposeOp;
        private static Operate GetTransposeOp() => _transposeOp ??= new Operate("Common/Transpose", "CSMain"
            , "src_height", "src_width", "from", "to");
        public static void Transpose(Tensor src, Tensor dst)
        {
            GetTransposeOp().CreateTask()
                .SetInt(src.Size[0])
                .SetInt(src.Size[1])
                .SetTensor(src)
                .SetTensor(dst)
                .Dispatch(dst.FlattenSize);
        }
        
    }
}