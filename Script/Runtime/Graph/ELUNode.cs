using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    
    public class ELUNode : IGraphNode
    {
        
        private static Operate _valueOp;
        
        private static Operate _derivativeOp;

        private static Operate GetValueOperate() => _valueOp ??= new Operate("ELU", "Value"
            , "r_buffer1", "rw_buffer1");

        private static Operate GetDerivativeOperate() => _derivativeOp ??= new Operate("ELU", "Derivative"
            , "gradient", "r_buffer1", "rw_buffer1");

        private readonly IGraphNode _input;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public ELUNode(IGraphNode input)
        {
            _input = input;
            var size = input.GetOutput().count;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }
        
        public ComputeBuffer GetOutput()
        {
            return _output;
        }

        public ComputeBuffer GetGradient()
        {
            return _gradient;
        }

        public void Calculate()
        {
            GetValueOperate().CreateTask()
                .SetBuffer(_input.GetOutput())
                .SetBuffer(_output)
                .Dispatch(new Vector3Int(_output.count, 1, 1));
        }

        public void GradientPropagation()
        {
            GetDerivativeOperate().CreateTask()
                .SetBuffer(_gradient)
                .SetBuffer(_output)
                .SetBuffer(_input.GetGradient())
                .Dispatch(new Vector3Int(_output.count, 1, 1));
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_input);
        }
    }
}