using System.Collections;
using System.Diagnostics;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.Util;
using UnityEngine;

namespace Blue.Samples
{
    public abstract class BaseSample
    {

        public const int BatchSize = 32;
        
        private readonly Model _model = new Model();
        private readonly Stopwatch _stopwatch = new Stopwatch();
        
        private ComputeBuffer _outputTarget;

        public int Epoch { get; private set; }

        public bool IsRunning { get; private set; }

        public int TrainCount { get; private set; }

        public int TestCount { get; private set; }

        public IGraphNode InputNode { get; private set; }

        public IGraphNode OutputNode { get; private set; }

        public abstract string Info { get; }

        public long BatchTrainingTime => _stopwatch.ElapsedMilliseconds;

        protected abstract int GetTrainCount();

        protected abstract void GetTrainData(int index, out float[] input, out float[] output);

        protected abstract int GetTestCount();

        protected abstract void GetTestData(int index, out float[] input, out float[] output);

        protected abstract void SetupGraph(out IGraphNode input, out IGraphNode output);

        public IEnumerator Run(int epochs)
        {
            if (IsRunning) yield break;
            IsRunning = true;
            SetupGraph(out var input, out var output);
            InputNode = input;
            OutputNode = output;
            _model.BatchSize = BatchSize;
            _model.Load(output, new AdamOptimizer());
            _outputTarget = new ComputeBuffer(OutputNode.GetOutput().count, 4);
            while (Epoch < epochs)
            {
                Epoch++;
                TrainCount = 0;
                OnEpochStart();
                yield return Train();
            }
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
                _stopwatch.Restart();
                for (var i = 0; i < BatchSize && TrainCount < GetTrainCount(); i++)
                {
                    OnTrainUpdate();
                }
                _stopwatch.Stop();
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
            LossFunction.CrossEntropyLoss(OutputNode, _outputTarget);
            _model.BackwardPropagation();
            _model.UpdateParams();
            TrainCount++;
            OnTrain(OutputNode.GetOutput(), _outputTarget);
        }

        private void OnTestUpdate()
        {
            GetTestData(TestCount, out var input, out var output);
            InputNode.GetOutput().SetData(input);
            _model.ForwardPropagation();
            _outputTarget.SetData(output);
            TestCount++;
            OnTest(OutputNode.GetOutput(), _outputTarget);
        }

        protected virtual void OnEpochStart()
        {
            
        }

        protected virtual void OnTrain(ComputeBuffer output, ComputeBuffer target)
        {
            
        }

        protected virtual void OnTest(ComputeBuffer output, ComputeBuffer target)
        {
            
        }

    }
}