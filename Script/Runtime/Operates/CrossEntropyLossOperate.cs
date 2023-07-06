using UnityEngine;

namespace Blue.Operates
{
    public class CrossEntropyLossOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("CrossEntropyLoss", "CSMain"
            , "r_buffer1", "r_buffer2", "rw_buffer1");
        
        public static void Calculate(ComputeBuffer p, ComputeBuffer y, ComputeBuffer gradient)
        {
            GetOperate().CreateTask()
                .SetBuffer(p)
                .SetBuffer(y)
                .SetBuffer(gradient)
                .Dispatch(new Vector3Int(p.count, 1, 1));
        }
    }
}