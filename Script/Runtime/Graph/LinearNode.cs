using System;
using System.Collections.Generic;
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
        private readonly List<OperateInstance> _forwardOpList = new List<OperateInstance>();
        private readonly List<OperateInstance> _backwardOpList = new List<OperateInstance>();
        
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
            Op.Clear(_tBias, 1f / batchSize).Dispatch().Destroy();
            UpdateOperate();
        }

        private void UpdateOperate()
        {
            foreach (var op in _forwardOpList)
            {
                op.Destroy();
            }
            _forwardOpList.Clear();
            _forwardOpList.Add(Op.MatMul(_input.GetOutput()
                , _weight.GetOutput()
                , _output));
            _forwardOpList.Add(Op.Increment(_output, _bias.GetOutput()));
            
            foreach (var op in _backwardOpList)
            {
                op.Destroy();
            }
            _backwardOpList.Clear();
            _backwardOpList.Add(Op.Transpose(_weight.GetOutput()
                , _tWeight));
            _backwardOpList.Add(Op.MatMul(_gradient
                , _tWeight
                , _input.GetGradient()));
            _backwardOpList.Add(Op.Transpose(_input.GetOutput()
                , _tInput));
            _backwardOpList.Add(Op.MatMul(_tInput
                , _gradient
                , _weight.GetGradient()));
            _backwardOpList.Add(Op.Translate(_weight.GetGradient(), 1f / _input.GetOutput().Size[0], 0f));
            _backwardOpList.Add(Op.MatMul(_tBias, _gradient, _bias.GetGradient()));
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
                _tBias.ResizeWithValue(1f / batchSize, 1, batchSize);
                UpdateOperate();
            }

            foreach (var op in _forwardOpList)
            {
                op.Dispatch();
            }
        }

        public void Backward()
        {
            foreach (var op in _backwardOpList)
            {
                op.Dispatch();
            }
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            _tWeight.Release();
            _tInput.Release();
            _tBias.Release();
            foreach (var op in _forwardOpList)
            {
                op.Destroy();
            }
            _forwardOpList.Clear();
            foreach (var op in _backwardOpList)
            {
                op.Destroy();
            }
            _backwardOpList.Clear();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
            action(_weight);
            action(_bias);
        }
    }
}