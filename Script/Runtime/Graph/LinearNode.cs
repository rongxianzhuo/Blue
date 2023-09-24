using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : BasicGraphNode
    {

        private readonly IGraphNode _input;
        private readonly IGraphNode _weight;
        private readonly IGraphNode _bias;
        private readonly Tensor _tInput;
        private readonly Tensor _tWeight;
        private readonly Tensor _tBias;
        
        public LinearNode(IGraphNode input, IGraphNode weight, IGraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _input = input;
            _weight = weight;
            _bias = bias;
            _tInput = input.GetOutput().Transpose();
            _tWeight = weight.GetOutput().Transpose();
            _tBias = new Tensor(1, batchSize);
            Op.Clear(_tBias, 1f / batchSize).Dispatch().Destroy();
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forwardOpList, List<Operate> backwardOpList)
        {
            _tInput.Resize(_input.GetOutput().Size[1], batchSize);
            _tBias.ResizeWithValue(1f / batchSize, 1, batchSize);
            
            forwardOpList.Add(Op.MatMul(_input.GetOutput()
                , _weight.GetOutput()
                , GetOutput()));
            forwardOpList.Add(Op.Increment(GetOutput(), _bias.GetOutput()));
            
            backwardOpList.Add(Op.Transpose(_weight.GetOutput()
                , _tWeight));
            backwardOpList.Add(Op.MatMul(GetGradient()
                , _tWeight
                , _input.GetGradient()));
            backwardOpList.Add(Op.Transpose(_input.GetOutput()
                , _tInput));
            backwardOpList.Add(Op.MatMul(_tInput
                , GetGradient()
                , _weight.GetGradient()));
            backwardOpList.Add(Op.Translate(_weight.GetGradient(), 1f / _input.GetOutput().Size[0], 0f));
            backwardOpList.Add(Op.MatMul(_tBias, GetGradient(), _bias.GetGradient()));
        }

        protected override void GetOutputSize(out int batchSize, out int size)
        {
            batchSize = _input.GetOutput().Size[0];
            size = _bias.GetOutput().FlattenSize;
        }

        protected override void OnDestroy()
        {
            _tWeight.Release();
            _tInput.Release();
            _tBias.Release();
        }

        public override void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
            action(_weight);
            action(_bias);
        }
    }
}