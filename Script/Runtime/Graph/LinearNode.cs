using System;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : IGraphNode
    {

        private readonly IGraphNode _input;
        private readonly IGraphNode _weight;
        private readonly IGraphNode _bias;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly Tensor _tInput;
        private readonly Tensor _tWeight;
        private readonly Tensor _tBias;
        
        public LinearNode(IGraphNode input, IGraphNode weight, IGraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _input = input;
            _weight = weight;
            _bias = bias;
            _output = new Tensor(batchSize, bias.GetOutput().FlattenSize);
            _gradient = new Tensor(batchSize, bias.GetOutput().FlattenSize);
            _tInput = new Tensor(input.GetOutput().Size[1], batchSize);
            _tWeight = new Tensor(weight.GetOutput().Size);
            _tBias = new Tensor(batchSize);
            Op.Clear(_tBias, 1f / batchSize);
        }

        public Tensor GetOutput() => _output;

        public Tensor GetGradient() => _gradient;

        public void Forward()
        {
            Op.MatMul(_input.GetOutput()
                , _input.GetOutput().Size[1]
                , _weight.GetOutput()
                , _output.Size[1]
                , _output);
            Op.Increment(_output, _bias.GetOutput());
        }

        public void Backward()
        {
            Op.Transpose(_weight.GetOutput()
                , _input.GetOutput().Size[1]
                , _output.Size[1]
                , _tWeight);
            Op.MatMul(_gradient
                , _gradient.Size[1]
                , _tWeight
                , _input.GetOutput().Size[1]
                , _input.GetGradient());
            Op.Transpose(_input.GetOutput()
                , _input.GetOutput().Size[0]
                , _input.GetOutput().Size[1]
                , _tInput);
            Op.MatMul(_tInput
                , _input.GetOutput().Size[0]
                , _gradient
                , _output.Size[1]
                , _weight.GetGradient());
            Op.Translate(_weight.GetGradient(), 1f / _input.GetOutput().Size[0], 0f);
            Op.MatMul(_tBias, _tBias.FlattenSize, _gradient, _output.Size[1], _bias.GetGradient());
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            _tWeight.Release();
            _tInput.Release();
            _tBias.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
            action(_weight);
            action(_bias);
        }
    }
}