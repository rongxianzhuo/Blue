using Blue.Graph;
using Blue.Kit;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.Runtime.NN
{
    
    public class Linear : Module
    {

        public readonly ComputationalNode Weight;
        public readonly ComputationalNode Bias;

        public Linear(int input, int output)
        {
            Weight = CreateParameter(input, output);
            Bias = CreateParameter(output);
            
            var min = -Mathf.Sqrt(1f / (input + output));
            var max = -min;
            var array = new float[Weight.FlattenSize];
            for (var i = 0; i < Weight.FlattenSize; i++)
            {
                array[i] = Random.Range(min, max);
            }
            Weight.SetData(array);
        }

        public override ComputationalNode Forward(params ComputationalNode[] input)
        {
            var node = input[0];
            var size = Bias.FlattenSize;
            var batchSize = node.Size[0];
            return new ComputationalNode(new []{node, Weight, Bias}, batchSize, size)
                .MatMul(node, Weight)
                .Add(Bias);
        }
    }
}