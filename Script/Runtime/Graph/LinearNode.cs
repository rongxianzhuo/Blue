using System;
using Blue.Core;

namespace Blue.Graph
{
    public class LinearNode : IGraphNode
    {
        
        private static Operate _forwardOp;
        private static Operate GetForwardOp() => _forwardOp ??= new Operate("Graph/Linear/Forward", "CSMain"
            , "input_count", "input_layer", "weight", "bias", "output");
        
        private static Operate _backwardInputOp;
        private static Operate GetBackwardInputOp() => _backwardInputOp ??= new Operate("Graph/Linear/BackwardInput", "CSMain"
            , "input_count", "output_count", "weight", "gradient", "input_gradient");
        
        private static Operate _backwardWeightOp;
        private static Operate GetBackwardWeightOp() => _backwardWeightOp ??= new Operate("Graph/Linear/BackwardWeight", "CSMain"
            , "input_count", "input_layer", "gradient", "weight_gradient");
        
        private static Operate _backwardBiasOp;
        private static Operate GetBackwardBiasOp() => _backwardBiasOp ??= new Operate("Graph/Linear/BackwardBias", "CSMain"
            , "gradient", "bias_gradient");

        private readonly IGraphNode _input;
        private readonly IGraphNode _weight;
        private readonly IGraphNode _bias;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        
        public LinearNode(IGraphNode input, IGraphNode weight, IGraphNode bias, int size)
        {
            _input = input;
            _weight = weight;
            _bias = bias;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
        }

        public Tensor GetOutput() => _output;

        public Tensor GetGradient() => _gradient;

        public void Forward()
        {
            GetForwardOp().CreateTask()
                .SetInt(_input.GetOutput().FlattenSize)
                .SetTensor(_input.GetOutput())
                .SetTensor(_weight.GetOutput())
                .SetTensor(_bias.GetOutput())
                .SetTensor(_output)
                .Dispatch(_output.FlattenSize);
        }

        public void Backward()
        {
            GetBackwardInputOp().CreateTask()
                .SetInt(_input.GetOutput().FlattenSize)
                .SetInt(_output.FlattenSize)
                .SetTensor(_weight.GetOutput())
                .SetTensor(_gradient)
                .SetTensor(_input.GetGradient())
                .Dispatch(_input.GetGradient().FlattenSize);
            GetBackwardWeightOp().CreateTask()
                .SetInt(_input.GetOutput().FlattenSize)
                .SetTensor(_input.GetOutput())
                .SetTensor(_gradient)
                .SetTensor(_weight.GetGradient())
                .Dispatch(_weight.GetGradient().FlattenSize);
            GetBackwardBiasOp().CreateTask()
                .SetTensor(_gradient)
                .SetTensor(_bias.GetGradient())
                .Dispatch(_bias.GetGradient().FlattenSize);
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
            action(_weight);
            action(_bias);
        }
    }
}