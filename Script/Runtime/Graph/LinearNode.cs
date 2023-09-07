using System;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : IGraphNode
    {
        
        private static Operate _backwardInputOp;
        private static Operate GetBackwardInputOp() => _backwardInputOp ??= new Operate("Graph/Linear/BackwardInput", "CSMain"
            , "input_count", "output_count", "weight", "gradient", "input_gradient");

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
            Op.MatMul(_input.GetOutput()
                , _input.GetOutput().FlattenSize
                , _weight.GetOutput()
                , _output.FlattenSize
                , _output);
            Op.Increment(_output, _bias.GetOutput());
        }

        public void Backward()
        {
            Op.MatMul(_weight.GetOutput()
                , _gradient.FlattenSize
                , _gradient
                , 1
                , _input.GetGradient());
            Op.MatMul(_input.GetOutput()
                , 1
                , _gradient
                , _output.FlattenSize
                , _weight.GetGradient());
            Op.Copy(_gradient, 0, _bias.GetGradient(), 0, _gradient.FlattenSize);
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