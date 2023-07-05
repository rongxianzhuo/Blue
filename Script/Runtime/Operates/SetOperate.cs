using UnityEngine;

namespace Blue.Operates
{
    public static class SetOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Set", "CSMain"
            , "f1", "rw_buffer1");

        public static void Calculate(ComputeBuffer buffer, float f)
        {
            GetOperate().CreateTask()
                .SetFloat(f)
                .SetBuffer(buffer)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
        
    }
}