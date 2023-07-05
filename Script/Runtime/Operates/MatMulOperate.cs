using UnityEngine;

namespace Blue.Operates
{
    public static class MatMulOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("MatMul", "CSMain"
            , "column"
            , "r_buffer1"
            , "r_buffer2"
            , "rw_buffer1");
        
        public static void Calculate(int column, ComputeBuffer left, ComputeBuffer right, ComputeBuffer result)
        {
            GetOperate().CreateTask()
                .SetInt(column)
                .SetBuffer(left)
                .SetBuffer(right)
                .SetBuffer(result)
                .Dispatch(new Vector3Int(right.count / column, left.count / column, 1));
        }
    }
}