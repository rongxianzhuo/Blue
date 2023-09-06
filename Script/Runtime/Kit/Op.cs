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
        public static void MatMul(Tensor left, int leftWidth, Tensor right, int rightWidth, Tensor result)
        {
            GetMatMulOp().CreateTask()
                .SetInt(leftWidth)
                .SetInt(rightWidth)
                .SetTensor(left)
                .SetTensor(right)
                .SetTensor(result)
                .Dispatch(result.FlattenSize);
        }
        
        private static Operate _incrementOp;
        private static Operate GetIncrementOp() => _incrementOp ??= new Operate("Common/Increment", "CSMain"
            , "r_buffer1", "rw_buffer1");
        public static void Increment(Tensor buffer, Tensor other)
        {
            GetIncrementOp().CreateTask()
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
        
    }
}