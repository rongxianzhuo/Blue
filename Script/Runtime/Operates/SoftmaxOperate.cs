using UnityEngine;

namespace Blue.Operates
{
    public class SoftmaxOperate
    {
        
        private static Operate _valueOp;
        
        private static Operate _derivativeOp;

        private static Operate GetValueOperate() => _valueOp ??= new Operate("Softmax", "Value"
            , "r_buffer", "rw_buffer");

        private static Operate GetDerivativeOperate() => _derivativeOp ??= new Operate("Softmax", "Derivative"
            , "r_buffer", "rw_buffer");

        public static void CalculateValue(ComputeBuffer input, ComputeBuffer output)
        {
            GetValueOperate().CreateTask()
                .SetBuffer(input)
                .SetBuffer(output)
                .Dispatch(new Vector3Int(input.count, 1, 1));
        }
        
        public static void CalculateDerivative(ComputeBuffer output, ComputeBuffer derivative)
        {
            GetDerivativeOperate().CreateTask()
                .SetBuffer(output)
                .SetBuffer(derivative)
                .Dispatch(new Vector3Int(output.count, 1, 1));
        }
    }
}