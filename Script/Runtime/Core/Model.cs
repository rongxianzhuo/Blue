using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Core
{

    public class Model : NodeGraph
    {

        private const int DefaultBatchSize = 32;

        private IOptimizer _optimizer;
        private int _batchSize = DefaultBatchSize;
        private int _paramsUpdateFlag = DefaultBatchSize;
        private int _requestBatchSize = DefaultBatchSize;
        private Operate _lossFunction;

        public bool IsTrainEnabled => _optimizer != null;

        public int BatchSize
        {
            set
            {
                if (_requestBatchSize == value) return;
                _requestBatchSize = Mathf.Max(1, value);
                if (IsTrainEnabled) return;
                _batchSize = _requestBatchSize;
                _paramsUpdateFlag = _requestBatchSize;
            }
        }

        public void EnableTrain(IOptimizer optimizer, string lossFunction)
        {
            DisableTrain();
            _batchSize = _requestBatchSize;
            _paramsUpdateFlag = _requestBatchSize;
            _optimizer = optimizer;
            _lossFunction = new Operate($"LossFunction/{lossFunction}", "CSMain"
                , "output", "target", "gradient");
        }

        public void DisableTrain()
        {
            _optimizer?.Destroy();
            _optimizer = null;
            _lossFunction = null;
        }

        public void Backward(ComputeBuffer target)
        {
            _lossFunction.CreateTask()
                .SetBuffer(Output.GetOutput())
                .SetBuffer(target)
                .SetBuffer(Output.GetGradient())
                .Dispatch(new Vector3Int(target.count, 1, 1));
            Backward();
            _paramsUpdateFlag--;
            if (_paramsUpdateFlag > 0) return;
            ForeachParameterNode(UpdateParameter);
            _batchSize = _requestBatchSize;
            _paramsUpdateFlag = _requestBatchSize;
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
            Op.Translate(node.TotalGradient, 1f / _batchSize, 0);
            _optimizer.Step(node.GetOutput(), node.TotalGradient);
            Op.Translate(node.TotalGradient, 0, 0);
        }
    }

}