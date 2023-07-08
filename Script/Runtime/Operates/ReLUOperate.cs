using UnityEngine;

namespace Blue.Operates
{
    public class ReLUOperate
    {
        
        private static Operate _valueOp;
        
        private static Operate _derivativeOp;

        private static Operate GetValueOperate() => _valueOp ??= new Operate("ReLU", "Value"
            , "r_buffer1", "rw_buffer1");

        private static Operate GetDerivativeOperate() => _derivativeOp ??= new Operate("ReLU", "Derivative"
            , "r_buffer1", "rw_buffer1");
        
        public static void CalculateValue(ComputeBuffer buffer, ComputeBuffer result)
        {
            GetValueOperate().CreateTask()
                .SetBuffer(buffer)
                .SetBuffer(result)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
        
        public static void CalculateDerivative(ComputeBuffer buffer, ComputeBuffer result)
        {
            GetDerivativeOperate().CreateTask()
                .SetBuffer(buffer)
                .SetBuffer(result)
                .Dispatch(new Vector3Int(buffer.count, 1, 1));
        }
    }
}