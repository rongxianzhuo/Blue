using UnityEngine;

namespace Blue.Operates
{
    public class SoftmaxOperate
    {
        
        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Softmax", "CSMain", "r_buffer", "rw_buffer");

        public static void Calculate(ComputeBuffer input, ComputeBuffer output)
        {
            GetOperate().CreateTask()
                .SetBuffer(input)
                .SetBuffer(output)
                .Dispatch(new Vector3Int(input.count, 1, 1));
        }
    }
}