using UnityEngine;

namespace Blue.Operates
{
    public class TransformOperate
    {
        
        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Transform", "CSMain"
            , "weight", "bias", "rw_buffer1");

        public static void Calculate(ComputeBuffer buffer, float weight, float bias)
        {
            GetOperate().CreateTask()
                .SetFloat(weight)
                .SetFloat(bias)
                .SetBuffer(buffer)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
    }
}