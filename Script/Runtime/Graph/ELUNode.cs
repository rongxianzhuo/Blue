using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    
    public class ELUNode : IGraphNode
    {

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
            ELUOperate.CalculateValue(_input.GetOutput(), _output);
        }

        public void GradientPropagation()
        {
            ELUOperate.CalculateDerivative(_output, _input.GetGradient());
            MulOperate.Calculate(_input.GetGradient(), _gradient);
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