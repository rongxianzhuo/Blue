using UnityEngine;

namespace Blue.Operates
{
    public static class XavierOperate
    {
        
        public static void WeightInit(ComputeBuffer weight, string activation, int inputCount, int outputCount)
        {
            var min = -1f;
            var max = 1f;
            switch (activation)
            {
                case "tanh":
                    min = -Mathf.Sqrt(6f / (inputCount + outputCount));
                    max = -min;
                    break;
                case "relu":
                    min = -Mathf.Sqrt(12f / (inputCount + outputCount));
                    max = -min;
                    break;
                case "sigmoid":
                    min = -Mathf.Sqrt(96f / (inputCount + outputCount));
                    max = -min;
                    break;
                default:
                    min = -Mathf.Sqrt(1f / (inputCount + outputCount));
                    max = -min;
                    break;
            }
            var array = new float[weight.count];
            for (var i = 0; i < array.Length; i++)
            {
                array[i] = Random.Range(min, max);
            }
            weight.SetData(array);
        }
    }
}