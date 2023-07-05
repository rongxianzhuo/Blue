using UnityEngine;

namespace Blue.Operates
{
    public static class DotOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Dot", "CSMain"
            , "buffer_size1"
            , "stride"
            , "interval"
            , "r_buffer1"
            , "r_buffer2"
            , "rw_buffer1");
        
        public static void Calculate(int stride, int interval, ComputeBuffer left, ComputeBuffer right, ComputeBuffer result)
        {
            GetOperate().CreateTask()
                .SetInt(left.count)
                .SetInt(stride)
                .SetInt(interval)
                .SetBuffer(left)
                .SetBuffer(right)
                .SetBuffer(result)
                .Dispatch(new Vector3Int(result.count, 1, 1));
        }
    }
}