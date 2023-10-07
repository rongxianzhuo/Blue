using System;
using Blue.Core;

namespace Blue.Graph
{
    public class DropoutNode : IGraphNode
    {

        private readonly float _dropout;
        private readonly IGraphNode _input;
        private readonly Tensor _weight;
        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private float[] _weightArray;
        private Operate _forward;
        private Operate _backward;
        
        public DropoutNode(IGraphNode input, float dropout)
        {
            _input = input;
            _dropout = dropout;
            _weightArray = new float[input.GetOutput().FlattenSize];
            _output = new Tensor(input.GetOutput().Size);
            _gradient = new Tensor(input.GetOutput().Size);
            _weight = new Tensor(input.GetOutput().Size);
        }

        public Tensor GetOutput()
        {
            return _output;
        }

        public Tensor GetGradient()
        {
            return _gradient;
        }

        public void Forward()
        {
            if (_input.GetOutput().FlattenSize != _output.FlattenSize)
            {
                _weightArray = new float[_input.GetOutput().FlattenSize];
                _output.Resize(_input.GetOutput().Size);
                _gradient.Resize(_input.GetOutput().Size);
                _weight.Resize(_input.GetOutput().Size);
                _forward?.Destroy();
                _backward?.Destroy();
                _forward = null;
                _backward = null;
            }

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

        public void Backward()
        {
            _backward.Dispatch();
            _input.Backward();
        }

        public void Destroy()
        {
            _weight.Release();
            _output.Release();
            _gradient.Release();
            _forward?.Destroy();
            _backward?.Destroy();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
        }
    }
}