using Blue.Core;

namespace Blue.Graph
{
    public class TensorNode : GraphNode
    {

        public readonly int Id;

        public readonly Tensor TotalGradient;

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public bool IsParameter => TotalGradient != null;

        public TensorNode(int id, bool isParam, params int[] size)
        {
            Id = id;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
            if (isParam)
            {
                BackwardOperates.Add(new Operate("Common/GradientIncrease", "CSMain")
                    .SetFloat("weight_decay", 0.000f)
                    .SetTensor("gradient", _gradient)
                    .SetTensor("weight", _output)
                    .SetTensor("total_gradient", TotalGradient)
                    .SetDispatchSize(TotalGradient.FlattenSize));
            }
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
            _output.Release();
            _gradient.Release();
            TotalGradient?.Release();
            base.OnDestroy();
        }
    }
}