using Blue.Core;
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
            Weight = new ComputationalNode(true, input, output);
            Bias = new ComputationalNode(true, output);
            RegisterParameter(Weight);
            RegisterParameter(Bias);
            
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
            var linearNode = new ComputationalNode(new []{node, Weight, Bias}, batchSize, size);
            
            // forward
            linearNode.AddForwardOperate(Op.MatMul(node
                , Weight
                , linearNode));
            linearNode.AddForwardOperate(Op.Add(linearNode, Bias));
            
            // input backward
            if (node.Gradient != null)
            {
                var tWeight = linearNode.CreateTempTensor(Weight.TransposeSize());
                linearNode.AddBackwardOperate(Op.Transpose(Weight
                    , tWeight));
                linearNode.AddBackwardOperate(Op.MatMul(linearNode.Gradient
                    , tWeight
                    , node.Gradient));
            }
            
            // weight backward
            var tInput = linearNode.CreateTempTensor(node.TransposeSize());
            linearNode.AddBackwardOperate(Op.Transpose(node
                , tInput));
            linearNode.AddBackwardOperate(Op.Translate(tInput, 1f / node.Size[0], 0f));
            linearNode.AddBackwardOperate(Op.IncreaseMatMul(tInput
                , linearNode.Gradient
                , Weight.Gradient));
            
            // bias
            linearNode.AddBackwardOperate(new Operate("Common/Add", "Backward")
                .SetInt("batch_size", batchSize)
                .SetInt("other_len", Bias.FlattenSize)
                .SetInt("result_len", linearNode.FlattenSize)
                .SetTensor("other_gradient", Bias.Gradient)
                .SetTensor("result_gradient", linearNode.Gradient)
                .SetDispatchSize(Bias.FlattenSize));

            return linearNode;
        }
    }
}