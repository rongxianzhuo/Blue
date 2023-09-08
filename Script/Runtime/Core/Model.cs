using Blue.Graph;
using Blue.Optimizers;

namespace Blue.Core
{

    public class Model : NodeGraph
    {

        private IOptimizer _optimizer;
        private Operate _lossFunction;

        public void EnableTrain(IOptimizer optimizer, string lossFunction)
        {
            DisableTrain();
            _optimizer = optimizer;
            _lossFunction = new Operate($"LossFunction/{lossFunction}", "CSMain"
                , "total_count", "output", "target", "gradient");
        }

        public void DisableTrain()
        {
            _optimizer?.Destroy();
            _optimizer = null;
            _lossFunction = null;
        }

        public void Backward(Tensor target)
        {
            _lossFunction.CreateTask()
                .SetInt(target.Size[1])
                .SetTensor(Output.GetOutput())
                .SetTensor(target)
                .SetTensor(Output.GetGradient())
                .Dispatch(target.FlattenSize);
            Backward();
            ForeachParameterNode(UpdateParameter);
        }

        public Model(IGraphNode outputNode) : base(outputNode)
        {
        }

        public override void Destroy()
        {
            DisableTrain();
            base.Destroy();
        }

        private void UpdateParameter(TensorNode node)
        {
            _optimizer.Step(node.GetOutput(), node.GetGradient());
        }
    }

}