using System;
using Blue.Core;

namespace Blue.Graph
{
    public class DropoutNode : GraphNode
    {

        private readonly float _dropout;
        private readonly GraphNode _input;
        private readonly Tensor _weight;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private float[] _weightArray;
        private Operate _forward;
        private Operate _backward;
        
        public DropoutNode(GraphNode input, float dropout)
        {
            InputNodes.Add(input);
            _input = input;
            _dropout = dropout;
            _weightArray = new float[input.GetOutput().FlattenSize];
            _output = new Tensor(input.GetOutput().Size);
            _gradient = new Tensor(input.GetOutput().Size);
            _weight = new Tensor(input.GetOutput().Size);
        }

        public override Tensor GetOutput()
        {
            return _output;
        }

        public override Tensor GetGradient()
        {
            return _gradient;
        }

        public override void Forward()
        {

            _forward ??= new Operate("Common/Mul", "CSMain")
                .SetTensor("a", _input.GetOutput())
                .SetTensor("b", _weight)
                .SetTensor("result", _output)
                .SetDispatchSize(_output.FlattenSize);
            _backward ??= new Operate("Common/Mul", "CSMain")
                .SetTensor("a", _gradient)
                .SetTensor("b", _weight)
                .SetTensor("result", _input.GetGradient())
                .SetDispatchSize(_input.GetGradient().FlattenSize);
            for (var i = 0; i < _weightArray.Length; i++)
            {
                _weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= _dropout ? 1f : 0f;
            }
            _weight.SetData(_weightArray);
            _forward.Dispatch();
        }

        public override void Backward()
        {
            _backward.Dispatch();
            _input.Backward();
        }

        public override void Destroy()
        {
            _weight.Release();
            _output.Release();
            _gradient.Release();
            _forward?.Destroy();
            _backward?.Destroy();
        }
    }
}