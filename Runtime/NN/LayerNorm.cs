using Blue.Core;
using Blue.Graph;
using Blue.Kit;
using UnityEngine;

namespace Blue.NN
{
    
    public class LayerNorm : Module
    {

        public readonly ComputationalNode Weight;
        public readonly ComputationalNode Bias;

        public LayerNorm(params int[] dim)
        {
            Weight = CreateParameter(dim);
            Bias = CreateParameter(dim);
            Op.Clear(Weight, 1f).Dispatch().Dispose();
            Op.Clear(Bias, 0f).Dispatch().Dispose();
        }

        public override ComputationalNode Build(params ComputationalNode[] input)
        {
            var node = input[0];
            var output = new ComputationalNode(new []{node, Weight, Bias}, node.Size);
            output.AddForwardOperate(new Operate("NN/LayerNorm", "Forward")
                .SetFloat("eps", 0.00001f)
                .SetInt("layer_size", Weight.FlattenSize)
                .SetTensor("a", node)
                .SetTensor("weight", Weight)
                .SetTensor("bias", Bias)
                .SetTensor("b", output)
                .SetDispatchSize(output.FlattenSize));
            output.AddBackwardOperate(new Operate("NN/LayerNorm", "BackwardWeight")
                .SetInt("layer_count", node.FlattenSize / Weight.FlattenSize)
                .SetInt("layer_size", Weight.FlattenSize)
                .SetTensor("b", output)
                .SetTensor("b_gradient", output.Gradient)
                .SetTensor("weight_gradient", Weight.Gradient)
                .SetDispatchSize(Weight.FlattenSize));
            output.AddBackwardOperate(new Operate("NN/LayerNorm", "BackwardBias")
                .SetInt("layer_count", node.FlattenSize / Bias.FlattenSize)
                .SetInt("layer_size", Weight.FlattenSize)
                .SetTensor("b_gradient", output.Gradient)
                .SetTensor("bias_gradient", Bias.Gradient)
                .SetDispatchSize(Bias.Gradient.FlattenSize));
            if (node.Gradient != null) output.AddBackwardOperate(new Operate("NN/LayerNorm", "BackwardInput")
                .SetFloat("eps", 0.00001f)
                .SetInt("layer_count", node.FlattenSize / Bias.FlattenSize)
                .SetInt("layer_size", Weight.FlattenSize)
                .SetTensor("a", node)
                .SetTensor("b_gradient", output.Gradient)
                .SetTensor("weight", Weight)
                .SetTensor("a_gradient", node.Gradient)
                .SetDispatchSize(Bias.Gradient.FlattenSize));
            return output;
        }
    }
}