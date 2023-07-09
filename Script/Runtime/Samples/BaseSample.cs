using System.Collections;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Samples
{
    public abstract class BaseSample
    {

        public const int BatchSize = 32;
        
        private readonly Model _model = new Model();
        
        private ComputeBuffer _outputTarget;

        public Model GetModel() => _model;

        public bool IsRunning { get; private set; }

        public int TrainCount { get; private set; }

        public int TestCount { get; private set; }

        public IGraphNode InputNode { get; private set; }

        public IGraphNode OutputNode { get; private set; }

        public abstract string Info { get; }

        protected abstract void OnTest(ComputeBuffer output, ComputeBuffer target);

        protected abstract int GetTrainCount();

        protected abstract void GetTrainData(int index, out float[] input, out float[] output);

        protected abstract int GetTestCount();

        protected abstract void GetTestData(int index, out float[] input, out float[] output);

        protected abstract void SetupGraph(out IGraphNode input, out IGraphNode output);

        public IEnumerator Run()
        {
            if (IsRunning) yield break;
            IsRunning = true;
            SetupGraph(out var input, out var output);
            InputNode = input;
            OutputNode = output;
            _model.Load(output, new AdamOptimizer(0.1f), BatchSize);
            _outputTarget = new ComputeBuffer(OutputNode.GetOutput().count, 4);
            yield return Train();
            yield return Test();
            Stop();
        }

        public void Stop()
        {
            if (!IsRunning) return;
            IsRunning = false;
            _model.Unload();
            _outputTarget.Release();
        }

        private IEnumerator Train()
        {
            while (IsRunning && TrainCount < GetTrainCount())
            {
                for (var i = 0; i < BatchSize && TrainCount < GetTrainCount(); i++)
                {
                    OnTrainUpdate();
                }
                yield return null;
            }
        }

        private IEnumerator Test()
        {
            while (IsRunning && TestCount < GetTestCount())
            {
                OnTestUpdate();
                yield return null;
            }
        }

        private void OnTrainUpdate()
        {
            GetTrainData(TrainCount, out var input, out var output);
            InputNode.GetOutput().SetData(input);
            _model.ForwardPropagation();
            _outputTarget.SetData(output);
            LossFunction.L2Loss(OutputNode, _outputTarget);
            _model.BackwardPropagation();
            _model.UpdateParams();
            TrainCount++;
        }

        private void OnTestUpdate()
        {
            GetTestData(TestCount, out var input, out var output);
            InputNode.GetOutput().SetData(input);
            _model.ForwardPropagation();
            TestCount++;
            _outputTarget.SetData(output);
            OnTest(OutputNode.GetOutput(), _outputTarget);
        }

    }
}