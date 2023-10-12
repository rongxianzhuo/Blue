using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class LinearNode : GraphNode
    {

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public LinearNode(GraphNode input, GraphNode weight, GraphNode bias)
        {
            var batchSize = input.GetOutput().Size[0];
            _output = CreateTensor(input.GetOutput().Size[0], bias.GetOutput().FlattenSize);
            _gradient = CreateTensor(_output.Size);
            var tInput = CreateTensor(input.GetOutput().TransposeSize());
            var tWeight = CreateTensor(weight.GetOutput().TransposeSize());
            var tBias = CreateTensor(1, batchSize);
            Op.Clear(tBias, 1f / batchSize).Dispatch().Destroy();
            InputNodes.Add(input);
            InputNodes.Add(weight);
            InputNodes.Add(bias);
            ForwardOperates.Add(Op.MatMul(input.GetOutput()
                , weight.GetOutput()
                , _output));
            ForwardOperates.Add(Op.Increment(_output, bias.GetOutput()));
            
            BackwardOperates.Add(Op.Transpose(weight.GetOutput()
                , tWeight));
            BackwardOperates.Add(Op.MatMul(_gradient
                , tWeight
                , input.GetGradient()));
            BackwardOperates.Add(Op.Transpose(input.GetOutput()
                , tInput));
            BackwardOperates.Add(Op.MatMul(tInput
                , _gradient
                , weight.GetGradient()));
            BackwardOperates.Add(Op.Translate(weight.GetGradient(), 1f / input.GetOutput().Size[0], 0f));
            BackwardOperates.Add(Op.MatMul(tBias, _gradient, bias.GetGradient()));
        }

        public override Tensor GetOutput()
        {
            return _output;
        }

        public override Tensor GetGradient()
        {
            return _gradient;
        }
    }
}