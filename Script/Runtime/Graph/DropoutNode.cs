using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public class DropoutNode : BasicGraphNode
    {

        private readonly float _dropout;
        private readonly GraphNode _input;
        private readonly Tensor _weight;
        private readonly float[] _weightArray;
        
        public DropoutNode(GraphNode input, float dropout)
        {
            InputNodes.Add(input);
            _input = input;
            _dropout = dropout;
            _weightArray = new float[input.GetOutput().FlattenSize];
            _weight = new Tensor(input.GetOutput().Size);
        }

        protected override void GetOutputSize(out int batchSize, out int size)
        {
            batchSize = _input.GetOutput().Size[0];
            size = _input.GetOutput().Size[1];
        }

        protected override void OnDestroy()
        {
            _weight.Release();
        }

        protected override void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward)
        {
            forward.Add(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", _input.GetOutput())
                .SetTensor("b", _weight)
                .SetTensor("result", GetOutput())
                .SetDispatchSize(GetOutput().FlattenSize));
            forward.Add(new Operate(() =>
            {
                for (var i = 0; i < _weightArray.Length; i++)
                {
                    _weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= _dropout ? 1f : 0f;
                }
                _weight.SetData(_weightArray);
            }));
            backward.Add(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", GetGradient())
                .SetTensor("b", _weight)
                .SetTensor("result", _input.GetGradient())
                .SetDispatchSize(_input.GetGradient().FlattenSize));
        }
    }
}