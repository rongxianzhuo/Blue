using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public static class Layer
    {

        public static IGraphNode DenseLayer(string name, IGraphNode input, int size, string activation=null)
        {
            var randomWeight = RandomWeight(size * input.GetOutput().count, activation, input.GetOutput().count, size);
            var weight = new TensorNode($"{name}.weight", true, randomWeight);
            var matMul = new MatMulNode(input, weight);
            var bias = new TensorNode($"{name}.bias", size, true);
            var add = OperateNode.Add(matMul, bias);
            return activation switch
            {
                "elu" => OperateNode.ELU(add),
                "relu" => OperateNode.ReLU(add),
                "sigmoid" => OperateNode.Sigmoid(add),
                _ => add
            };
        }
        
        private static float[] RandomWeight(int size, string activation, int inputCount, int outputCount)
        {
            float min;
            float max;
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
            var array = new float[size];
            for (var i = 0; i < array.Length; i++)
            {
                array[i] = Random.Range(min, max);
            }

            return array;
        }
        
    }
}