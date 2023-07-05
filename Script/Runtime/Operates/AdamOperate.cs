using UnityEngine;

namespace Blue.Operates
{
    public class AdamOperate
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Adam", "CSMain"
            , "t", "beta1", "beta2", "learning_rate", "g", "m", "v", "theta");

        public static void Calculate(float t
            , float beta1
            , float beta2
            , float learningRate
            , ComputeBuffer gradient
            , ComputeBuffer m
            , ComputeBuffer v
            , ComputeBuffer param)
        {
            GetOperate().CreateTask()
                .SetFloat(t)
                .SetFloat(beta1)
                .SetFloat(beta2)
                .SetFloat(learningRate)
                .SetBuffer(gradient)
                .SetBuffer(m)
                .SetBuffer(v)
                .SetBuffer(param)
                .Dispatch(new Vector3Int(param.count, 1, 1));
        }
    }
}