using UnityEngine;

namespace Blue.Operates
{
    public class CopyOperate
    {
        
        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Copy", "CSMain"
            , "r_buffer1", "rw_buffer1");

        public static void Calculate(ComputeBuffer src, ComputeBuffer dst)
        {
            GetOperate().CreateTask()
                .SetBuffer(src)
                .SetBuffer(dst)
                .Dispatch(new Vector3Int(dst.count, 1, 1));
        }
    }
}