using Blue.Core;
using UnityEngine;

namespace Blue.Kit
{
    public static class Op
    {
        
        private static Operate _translateOp;
        private static Operate GetTranslateOp() => _translateOp ??= new Operate("Common/Translate", "CSMain"
            , "weight", "bias", "rw_buffer1");
        public static void Translate(ComputeBuffer buffer, float weight, float bias)
        {
            GetTranslateOp().CreateTask()
                .SetFloat(weight)
                .SetFloat(bias)
                .SetBuffer(buffer)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }

        private static Operate _matMulOp;
        private static Operate GetMatMulOp() => _matMulOp ??= new Operate("Common/MatMul", "CSMain"
            , "wl"
            , "wr"
            , "left"
            , "right"
            , "result");
        public static void MatMul(ComputeBuffer left, int leftWidth, ComputeBuffer right, int rightWidth, ComputeBuffer result)
        {
            GetMatMulOp().CreateTask()
                .SetInt(leftWidth)
                .SetInt(rightWidth)
                .SetBuffer(left)
                .SetBuffer(right)
                .SetBuffer(result)
                .Dispatch(new Vector3Int(result.count, 1, 1));
        }
        
        private static Operate _incrementOp;
        private static Operate GetIncrementOp() => _incrementOp ??= new Operate("Common/Increment", "CSMain"
            , "r_buffer1", "rw_buffer1");
        public static void Increment(ComputeBuffer buffer, ComputeBuffer other)
        {
            GetIncrementOp().CreateTask()
                .SetBuffer(other)
                .SetBuffer(buffer)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
        
        private static Operate _copyOp;
        private static Operate GetCopyOp() => _copyOp ??= new Operate("Common/Copy", "CSMain"
            , "r_buffer1", "src_offset", "rw_buffer1", "dst_offset");

        public static void Copy(ComputeBuffer src, int srcStartIndex, ComputeBuffer dst, int dstStartIndex, int length)
        {
            GetCopyOp().CreateTask()
                .SetBuffer(src)
                .SetInt(srcStartIndex)
                .SetBuffer(dst)
                .SetInt(dstStartIndex)
                .Dispatch(new Vector3Int(length, 1, 1));
        }
        
    }
}