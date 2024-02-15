using System.Collections.Generic;
using Blue.Core;
using Blue.Data;
using Blue.Graph;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Kit
{
    public class DqnBrain : System.IDisposable
    {
        
        public interface IEnv
        {
            void GetState(float[] state);
            bool Update(int action, out float reward);
        }
        
        private const int BatchSize = 32;
        private const int ReplayBufferSize = 1024;

        private readonly List<float[]> _inputReplay = new List<float[]>();
        private readonly List<float[]> _targetReplay = new List<float[]>();

        private readonly int _actionSize;
        private readonly ComputationalGraph _runtimeDqn;
        private readonly ComputationalNode _runtimeInput;
        private readonly ComputationalGraph _trainDqn;
        private readonly ComputationalNode _trainInput;
        private readonly Operate _loss;
        private readonly Tensor _target;
        private readonly IOptimizer _optimizer;
        private readonly DatasetLoader _datasetLoader;
        private readonly float[] _tempState;

        private int _replayIndex;

        public DqnBrain(int stateSize, int actionSize)
        {
            _actionSize = actionSize;
            _tempState = new float[stateSize];
            
            _runtimeInput = new ComputationalNode(false, 1, stateSize);
            var runtimeOutput = _runtimeInput.Linear(128).Activation("relu").Linear(actionSize);
            _runtimeDqn = new ComputationalGraph(runtimeOutput);
            
            _trainInput = new ComputationalNode(false, BatchSize, stateSize);
            var trainOutput = _trainInput.Linear(128).Activation("relu").Linear(actionSize);
            _trainDqn = new ComputationalGraph(trainOutput);
            _target = new Tensor(BatchSize, actionSize);
            _loss = Op.L2Loss(_trainDqn.Output, _target, _trainDqn.Output.Gradient);
            _optimizer = new AdamOptimizer(_trainDqn.ParameterNodes);
            _datasetLoader = new DatasetLoader(BatchSize, BatchSize * ReplayBufferSize);

            for (var i = 0; i < BatchSize * ReplayBufferSize; i++)
            {
                _inputReplay.Add(new float[stateSize]);
                _targetReplay.Add(new float[actionSize]);
            }
        }

        public void TrainUpdate(IEnv trainEnv, IEnv previewEnv)
        {
            for (var i = 0; i < 128; i++)
            {
                if (_replayIndex >= ReplayBufferSize * BatchSize) break;
                var state = _inputReplay[_replayIndex];
                var qArray = _targetReplay[_replayIndex];
                _replayIndex++;
                trainEnv.GetState(state);
                _runtimeInput.SetData(state);
                _runtimeDqn.Forward();
                _runtimeDqn.Output.GetData(qArray);
                _runtimeDqn.Output.Max(out var action);
                if (Random.Range(0f, 1f) < 0.1f) action = Random.Range(0, _actionSize);
                if (trainEnv.Update(action, out var reward))
                {
                    trainEnv.GetState(_tempState);
                    _runtimeInput.SetData(_tempState);
                    _runtimeDqn.Forward();
                    reward += 0.9f * _runtimeDqn.Output.Max(out _);
                }
                qArray[action] = reward;
            }
            
            if (previewEnv != null)
            {
                previewEnv.GetState(_tempState);
                _runtimeInput.SetData(_tempState);
                _runtimeDqn.Forward();
                _runtimeDqn.Output.Max(out var action);
                previewEnv.Update(action, out _);
            }

            if (_replayIndex < ReplayBufferSize * BatchSize) return;
            _replayIndex = 0;
            
            _datasetLoader.Shuffle();
            _datasetLoader.LoadDataset(_inputReplay, _trainInput);
            _datasetLoader.LoadDataset(_targetReplay, _target);
            
            for (var i = 0; i < _datasetLoader.BatchCount; i++)
            {
                _datasetLoader.LoadBatch(i);
                _trainDqn.Forward();
                _trainDqn.ClearGradient();
                _loss.Dispatch();
                _trainDqn.Backward();
                _optimizer.Step();
            }
            
            _trainDqn.CopyParameterTo(_runtimeDqn);
        }

        public void Dispose()
        {
            _runtimeDqn.DisposeNodes();
            _trainDqn.DisposeNodes();
            _loss.Dispose();
            _target.Dispose();
            _optimizer.Dispose();
            _datasetLoader.Dispose();
        }
    }
}