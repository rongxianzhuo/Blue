using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public static class Layer
    {

        public static IGraphNode DenseLayer(string name, IGraphNode input, int size, string activation)
        {
            var weight = new TensorNode($"{name}.weight", size * input.GetOutput().count, true);
            WeightInit(weight.GetOutput(), activation, input.GetOutput().count, size);
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
        
        private static void WeightInit(ComputeBuffer weight, string activation, int inputCount, int outputCount)
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