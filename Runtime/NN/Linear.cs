using Blue.Graph;
using Blue.Kit;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.NN
{
    
    public class Linear : Module
    {

        public readonly ComputationalNode Weight;
        public readonly ComputationalNode Bias;

        public Linear(int input, int output)
        {
            Weight = CreateParameter(input, output);
            Bias = CreateParameter(output);
            
            var min = -1f / Mathf.Sqrt(input);
            var max = -min;
            Weight.SetData(array =>
            {
                for (var i = 0; i < array.Length; i++)
                {
                    array[i] = Random.Range(min, max);
                }
            });
            Bias.SetData(array =>
            {
                for (var i = 0; i < array.Length; i++)
                {
                    array[i] = Random.Range(min, max);
                }
            });
        }

        public override ComputationalNode Build(params ComputationalNode[] input)
        {
            return input[0].MatMul(Weight).AddInPlace(Bias);
        }
    }
}