using UnityEngine;

namespace Blue.Operates
{
    public static class MulOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Mul", "CSMain"
            , "r_buffer1", "rw_buffer1");
        
        public static void Calculate(ComputeBuffer buffer, ComputeBuffer other)
        {
            GetOperate().CreateTask()
                .SetBuffer(other)
                .SetBuffer(buffer)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
    }
}