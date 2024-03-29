using System;
using Blue.Core;
using Blue.Graph;

namespace Blue.NN
{
    public class Activation : Module
    {

        public static readonly Activation ReLU = new Activation("relu");

        public static readonly Activation Tanh = new Activation("tanh");

        private readonly string _activationName;

        public Activation(string activationName)
        {
            _activationName = activationName;
        }
        
        public override ComputationalNode Build(params ComputationalNode[] input)
        {
            var node = input[0];
            var shaderName = _activationName switch
            {
                "relu" => "Activation/ReLU",
                "elu" => "Activation/ELU",
                "sigmoid" => "Activation/Sigmoid",
                "tanh" => "Activation/Tanh",
                _ => throw new Exception("Unknown activation name")
            };
            var activation = new ComputationalNode(new[] { node }, node.Size);
            
            activation.AddForwardOperate(new Operate(shaderName, "Forward")
                .SetTensor("rw_output", activation)
                .SetTensor("input", node)
                .SetDispatchSize(node.FlattenSize));
            
            if (node.Gradient != null) activation.AddBackwardOperate(new Operate(shaderName, "Backward_input")
                .SetTensor("r_output", activation)
                .SetTensor("input_gradient", node.Gradient)
                .SetTensor("output_gradient", activation.Gradient)
                .SetDispatchSize(node.Gradient.FlattenSize));

            return activation;
        }
    }
}