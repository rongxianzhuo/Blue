using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : GraphNode
    {

        private readonly Tensor _output;
        private readonly Tensor _gradient;
        private readonly Tensor _tInput;
        private readonly Tensor _tWeight;
        private readonly Tensor _tBias;
        
        public LinearNode(GraphNode input, GraphNode weight, GraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _output = new Tensor(input.GetOutput().Size[0], bias.GetOutput().FlattenSize);
            _gradient = new Tensor(_output.Size);
            _tInput = input.GetOutput().Transpose();
            _tWeight = weight.GetOutput().Transpose();
            _tBias = new Tensor(1, batchSize);
            Op.Clear(_tBias, 1f / batchSize).Dispatch().Destroy();
            InputNodes.Add(input);
            InputNodes.Add(weight);
            InputNodes.Add(bias);
            ForwardOperates.Add(Op.MatMul(input.GetOutput()
                , weight.GetOutput()
                , _output));
            ForwardOperates.Add(Op.Increment(_output, bias.GetOutput()));
            
            BackwardOperates.Add(Op.Transpose(weight.GetOutput()
                , _tWeight));
            BackwardOperates.Add(Op.MatMul(_gradient
                , _tWeight
                , input.GetGradient()));
            BackwardOperates.Add(Op.Transpose(input.GetOutput()
                , _tInput));
            BackwardOperates.Add(Op.MatMul(_tInput
                , _gradient
                , weight.GetGradient()));
            BackwardOperates.Add(Op.Translate(weight.GetGradient(), 1f / input.GetOutput().Size[0], 0f));
            BackwardOperates.Add(Op.MatMul(_tBias, _gradient, bias.GetGradient()));
        }

        public override Tensor GetOutput()
        {
            return _output;
        }

        public override Tensor GetGradient()
        {
            return _gradient;
        }

        protected override void OnDestroy()
        {
            _tWeight.Release();
            _tInput.Release();
            _tBias.Release();
            _output.Release();
            _gradient.Release();
        }
    }
}