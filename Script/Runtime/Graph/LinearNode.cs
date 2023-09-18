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

        private OperateInstance _matMulForward;
        private OperateInstance _matMulBackward1;
        private OperateInstance _matMulBackward2;
        private OperateInstance _matMulBackward3;
        
        public LinearNode(IGraphNode input, IGraphNode weight, IGraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _input = input;
            _weight = weight;
            _bias = bias;
            _output = new Tensor(batchSize, bias.GetOutput().FlattenSize);
            _gradient = new Tensor(batchSize, bias.GetOutput().FlattenSize);
            _tInput = input.GetOutput().Transpose();
            _tWeight = weight.GetOutput().Transpose();
            _tBias = new Tensor(1, batchSize);
            Op.Clear(_tBias, 1f / batchSize);
            UpdateOperate();
        }

        private void UpdateOperate()
        {
            _matMulForward?.Destroy();
            _matMulForward = Op.MatMul(_input.GetOutput()
                , _weight.GetOutput()
                , _output);
            _matMulBackward1?.Destroy();
            _matMulBackward1 = Op.MatMul(_gradient
                , _tWeight
                , _input.GetGradient());
            _matMulBackward2?.Destroy();
            _matMulBackward2 = Op.MatMul(_tInput
                , _gradient
                , _weight.GetGradient());
            _matMulBackward3?.Destroy();
            _matMulBackward3 = Op.MatMul(_tBias, _gradient, _bias.GetGradient());
        }

        public Tensor GetOutput() => _output;

        public Tensor GetGradient() => _gradient;

        public void Forward()
        {
            var batchSize = _input.GetOutput().Size[0];
            if (batchSize != _output.Size[0])
            {
                var outputSize = _bias.GetOutput().FlattenSize;
                _output.Resize(batchSize, outputSize);
                _gradient.Resize(batchSize, outputSize);
                _tInput.Resize(_input.GetOutput().Size[1], batchSize);
                _tBias.Resize(1, batchSize);
                Op.Clear(_tBias, 1f / batchSize);
                UpdateOperate();
            }
            _matMulForward.Dispatch();
            Op.Increment(_output, _bias.GetOutput());
        }

        public void Backward()
        {
            Op.Transpose(_weight.GetOutput()
                , _tWeight);
            _matMulBackward1.Dispatch();
            Op.Transpose(_input.GetOutput()
                , _tInput);
            _matMulBackward2.Dispatch();
            Op.Translate(_weight.GetGradient(), 1f / _input.GetOutput().Size[0], 0f);
            _matMulBackward3.Dispatch();
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            _tWeight.Release();
            _tInput.Release();
            _tBias.Release();
            _matMulForward?.Destroy();
            _matMulBackward1?.Destroy();
            _matMulBackward2?.Destroy();
            _matMulBackward3?.Destroy();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
            action(_weight);
            action(_bias);
        }
    }
}