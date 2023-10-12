using Blue.Core;

namespace Blue.Graph
{
    public class DropoutNode : GraphNode
    {

        private readonly Tensor _output;
        private readonly Tensor _gradient;
        
        public DropoutNode(GraphNode input, float dropout)
        {
            _output = CreateTensor(input.GetOutput().Size);
            _gradient = CreateTensor(_output.Size);
            InputNodes.Add(input);
            var weightArray = new float[input.GetOutput().FlattenSize];
            var weight = CreateTensor(input.GetOutput().Size);
            ForwardOperates.Add(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", input.GetOutput())
                .SetTensor("b", weight)
                .SetTensor("result", _output)
                .SetDispatchSize(_output.FlattenSize));
            ForwardOperates.Add(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= dropout ? 1f : 0f;
                }
                weight.SetData(weightArray);
            }));
            BackwardOperates.Add(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", _gradient)
                .SetTensor("b", weight)
                .SetTensor("result", input.GetGradient())
                .SetDispatchSize(input.GetGradient().FlattenSize));
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