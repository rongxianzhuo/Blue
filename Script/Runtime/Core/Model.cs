using Blue.Graph;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Core
{

    public class Model : NodeGraph
    {
        
        private static Operate _translateOperate;

        private static Operate GetTranslateOperate() => _translateOperate ??= new Operate("Common/Translate", "CSMain"
            , "weight", "bias", "rw_buffer1");

        private const int DefaultBatchSize = 32;

        private IOptimizer _optimizer;
        private int _batchSize = DefaultBatchSize;
        private int _paramsUpdateFlag = DefaultBatchSize;
        private int _requestBatchSize = DefaultBatchSize;
        private Operate _lossFunction;

        public bool IsEnableTrain => _optimizer != null;

        public int BatchSize
        {
            set
            {
                if (_requestBatchSize == value) return;
                _requestBatchSize = Mathf.Max(1, value);
                if (IsEnableTrain) return;
                _batchSize = _requestBatchSize;
                _paramsUpdateFlag = _requestBatchSize;
            }
        }

        public void EnableTrain(IOptimizer optimizer, string lossFunction)
        {
            DisableTrain();
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

        public void UpdateParams()
        {
            _paramsUpdateFlag--;
            var shouldUpdateParams = _paramsUpdateFlag <= 0;
            if (shouldUpdateParams) _paramsUpdateFlag = _requestBatchSize;
            else return;
            ForeachParameterNode(UpdateParameter);
            _batchSize = _requestBatchSize;
        }

        public void BackwardPropagation(ComputeBuffer target)
        {
            _lossFunction.CreateTask()
                .SetBuffer(Output.GetOutput())
                .SetBuffer(target)
                .SetBuffer(Output.GetGradient())
                .Dispatch(new Vector3Int(target.count, 1, 1));
            Backward();
        }

        public Model(IGraphNode outputNode) : base(outputNode)
        {
        }

        public override void Destroy()
        {
            DisableTrain();
            base.Destroy();
        }

        private void UpdateParameter(DataNode node)
        {
            GetTranslateOperate().CreateTask()
                .SetFloat(1f / _batchSize)
                .SetFloat(0)
                .SetBuffer(node.TotalGradient)
                .Dispatch(new Vector3Int(node.GetOutput().count, 1, 1));
            _optimizer.OnBackwardPropagation(node.GetOutput(), node.TotalGradient);
            GetTranslateOperate().CreateTask()
                .SetFloat(0)
                .SetFloat(0)
                .SetBuffer(node.TotalGradient)
                .Dispatch(new Vector3Int(node.TotalGradient.count, 1, 1));
        }
    }

}