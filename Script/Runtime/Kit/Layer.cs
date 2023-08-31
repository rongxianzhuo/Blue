using System.Collections.Generic;
using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public static class Layer
    {

        public static IGraphNode DenseLayer(string name, IGraphNode input, int size, string activation=null)
        {
            var randomWeight = new float[size * input.GetOutput().count];
            var weight = new TensorNode($"{name}.weight", randomWeight.Length, true);
            weight.GetOutput().GetData(randomWeight); // make it sync
            RandomWeight(randomWeight, activation, input.GetOutput().count, size);
            weight.GetOutput().SetData(randomWeight);
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
        
        private static void RandomWeight(IList<float> weight, string activation, int inputCount, int outputCount)
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
            for (var i = 0; i < weight.Count; i++)
            {
                weight[i] = Random.Range(min, max);
            }
        }
        
    }
}