using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : BasicGraphNode
    {

        private readonly GraphNode _input;
        private readonly GraphNode _weight;
        private readonly GraphNode _bias;
        private readonly Tensor _tInput;
        private readonly Tensor _tWeight;
        private readonly Tensor _tBias;
        
        public LinearNode(GraphNode input, GraphNode weight, GraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _input = input;
            _weight = weight;
            _bias = bias;
            _tInput = input.GetOutput().Transpose();
            _tWeight = weight.GetOutput().Transpose();
            _tBias = new Tensor(1, batchSize);
            Op.Clear(_tBias, 1f / batchSize).Dispatch().Destroy();
            InputNodes.Add(input);
            InputNodes.Add(weight);
            InputNodes.Add(bias);
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forwardOpList, List<Operate> backwardOpList)
        {
            
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
    }
}