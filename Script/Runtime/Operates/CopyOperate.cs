using UnityEngine;

namespace Blue.Operates
{
    public class CopyOperate
    {
        
        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Copy", "CSMain"
            , "r_buffer1", "src_offset", "rw_buffer1", "dst_offset");

        public static void Calculate(ComputeBuffer src, int srcOffset, ComputeBuffer dst, int dstOffset)
        {
            GetOperate().CreateTask()
                .SetBuffer(src)
                .SetInt(srcOffset)
                .SetBuffer(dst)
                .SetInt(dstOffset)
                .Dispatch(new Vector3Int(dst.count, 1, 1));
        }
    }
}