using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.NN;
using UnityEngine;
using Random = System.Random;

namespace Blue.RL
{
    public class DqnAgent : System.IDisposable
    {
        
        public interface IEnv
        {
            void Reset();
            void GetState(float[] state);
            bool Update(int action, out float reward);
        }

        private class Dqn : System.IDisposable
        {
            
            public readonly Module Model;
            public readonly ComputationalNode RuntimeInput;
            public readonly ComputationalNode TrainInput;
            public readonly ComputationalGraph RuntimeGraph;
            public readonly ComputationalGraph TrainGraph;

            public Dqn(int stateSize, int batchSize, Module qNetwork)
            {
                Model = qNetwork;
                RuntimeInput = new ComputationalNode(false, 1, stateSize);
                RuntimeGraph = qNetwork.Build(RuntimeInput).Graph();
                TrainInput = new ComputationalNode(false, batchSize, stateSize);
                TrainGraph = qNetwork.Build(TrainInput).Graph();
            }

            public void Dispose()
            {
                RuntimeGraph.Dispose();
                TrainGraph.Dispose();
            }
        }

        private class ReplayMemory
        {

            public readonly int Capacity;
            public readonly int StateSize;
            public readonly int ActionSize;

            private readonly Random _rand;
            
            private readonly List<float[]> _stateReplay = new List<float[]>();
            private readonly List<float[]> _actionReplay = new List<float[]>();
            private readonly List<float[]> _rewardReplay = new List<float[]>();
            private readonly List<float[]> _nextStateReplay = new List<float[]>();
            private readonly List<float[]> _doneReplay = new List<float[]>();
            
            private readonly List<float[]> _sampleState = new List<float[]>();
            private readonly List<float[]> _sampleAction = new List<float[]>();
            private readonly List<float[]> _sampleReward = new List<float[]>();
            private readonly List<float[]> _sampleNextState = new List<float[]>();
            private readonly List<float[]> _sampleDone = new List<float[]>();

            private int _nextIndex;

            public int Size => _stateReplay.Count;

            public ReplayMemory(int capacity, int stateSize, int actionSize, Random rand)
            {
                _rand = rand;
                Capacity = capacity;
                StateSize = stateSize;
                ActionSize = actionSize;
            }

            public void Push(out float[] state, out float[] action, out float[] reward, out float[] nextState, out float[] done)
            {
                if (Size < Capacity)
                {
                    _stateReplay.Add(new float[StateSize]);
                    _actionReplay.Add(new float[ActionSize]);
                    _rewardReplay.Add(new float[ActionSize]);
                    _nextStateReplay.Add(new float[StateSize]);
                    _doneReplay.Add(new float[ActionSize]);
                }

                state = _stateReplay[_nextIndex];
                action = _actionReplay[_nextIndex];
                reward = _rewardReplay[_nextIndex];
                nextState = _nextStateReplay[_nextIndex];
                done = _doneReplay[_nextIndex];
                _nextIndex = (_nextIndex + 1) % Capacity;
            }

            public void SampleBatch(int batchSize
                , out IReadOnlyList<float[]> state
                , out IReadOnlyList<float[]> action
                , out IReadOnlyList<float[]> reward
                , out IReadOnlyList<float[]> nextState
                , out IReadOnlyList<float[]> done)
            {
                state = _sampleState;
                action = _sampleAction;
                reward = _sampleReward;
                nextState = _sampleNextState;
                done = _sampleDone;
                
                _sampleState.Clear();
                _sampleAction.Clear();
                _sampleReward.Clear();
                _sampleNextState.Clear();
                _sampleDone.Clear();
                
                while (batchSize-- > 0)
                {
                    var i = _rand.Next(0, _stateReplay.Count);
                    _sampleState.Add(_stateReplay[i]);
                    _sampleAction.Add(_actionReplay[i]);
                    _sampleReward.Add(_rewardReplay[i]);
                    _sampleNextState.Add(_nextStateReplay[i]);
                    _sampleDone.Add(_doneReplay[i]);
                }
            }
        }

        private readonly int _totalTrainSteps;
        private readonly Dqn _dqn;
        private readonly Dqn _targetDqn;
        private readonly SmoothL1Loss _loss;
        private readonly IOptimizer _optimizer;
        private readonly Tensor _rewardTensor;
        private readonly Tensor _doneTensor;
        private readonly Tensor _actionTensor;
        private readonly List<Operate> _targetQOp = new List<Operate>();
        private readonly OperateList _updateTargetOp = new OperateList();
        private readonly OperateList _clipGradNormOp = new OperateList();
        private readonly ReplayMemory _memory;
        private readonly int _batchSize;
        private readonly int _learningStarts = 100;
        private readonly int _trainFreq = 4;
        private readonly Random _rand = new Random();

        private uint _trainStep;
        
        public bool IsTrainCompleted => _trainStep >=  _totalTrainSteps;

        public DqnAgent(int stateSize
            , int actionSize
            , int batchSize
            , int replayBufferSize
            , int totalTrainSteps
            , Module qNetwork
            , Module targetQNetwork)
        {
            _totalTrainSteps = totalTrainSteps;
            _batchSize = batchSize;
            _memory = new ReplayMemory(replayBufferSize, stateSize, actionSize, _rand);

            _rewardTensor = new Tensor(new []{batchSize, actionSize});
            _doneTensor = new Tensor(new []{batchSize, actionSize});
            _actionTensor = new Tensor(new []{batchSize, actionSize});

            _dqn = new Dqn(stateSize, batchSize, qNetwork);
            _targetDqn = new Dqn(stateSize, batchSize, targetQNetwork);
            
            _loss = new SmoothL1Loss(_dqn.TrainGraph.Output, _targetDqn.TrainGraph.Output, scale:actionSize);
            _optimizer = new AdamOptimizer(qNetwork.GetAllParameters(), learningRate:0.001f);
            _targetQOp.Add(new Operate("Common/Translate", "CSMain")
                .SetTensor("weight", _doneTensor)
                .SetTensor("bias", _rewardTensor)
                .SetTensor("buffer", _targetDqn.TrainGraph.Output)
                .SetDispatchSize(_targetDqn.TrainGraph.Output.FlattenSize));
            _targetQOp.Add(Op.Lerp(_targetDqn.TrainGraph.Output, _dqn.TrainGraph.Output, _actionTensor));
            _targetDqn.Model.CopyParameter(_dqn.Model, _updateTargetOp);
            foreach (var node in _dqn.Model.GetAllParameters())
            {
                _clipGradNormOp.Add(Op.ClipNorm(node.Gradient, 10f));
            }
        }

        public void TrainStep(IEnv trainEnv
            , float futureRewardDiscount = 0.99f
            , uint targetUpdateInterval = 10000)
        {
            if (IsTrainCompleted) return;
            _memory.Push(out var state, out var action, out var reward, out var nextState, out var done);
            trainEnv.GetState(state);
            _dqn.RuntimeInput.SetData(state);
            _dqn.RuntimeGraph.Forward();
            var a = SelectAction(_dqn.RuntimeGraph.Output, false);
            var d = trainEnv.Update(a, out var r);
            if (d) trainEnv.Reset();
            else trainEnv.GetState(nextState);
            for (var j = 0; j < reward.Length; j++)
            {
                reward[j] = r;
                done[j] = d ? 0f : futureRewardDiscount;
                action[j] = j == a ? 0 : 1;
            }
            _trainStep++;
            if (_trainStep < _learningStarts) return;
            if (_trainStep % _trainFreq != 0) return;
                
            _memory.SampleBatch(_batchSize
                , out var sampleState
                , out var sampleAction
                , out var sampleReward
                , out var sampleNextState
                , out var sampleDone);
            _dqn.TrainInput.SetData(sampleState);
            _actionTensor.SetData(sampleAction);
            _rewardTensor.SetData(sampleReward);
            _targetDqn.TrainInput.SetData(sampleNextState);
            _doneTensor.SetData(sampleDone);
            _dqn.TrainGraph.Forward();
            _targetDqn.TrainGraph.Forward();
            foreach (var o in _targetQOp) o.Dispatch();
            _dqn.TrainGraph.ClearGradient();
            _loss.Backward();
            _clipGradNormOp.Dispatch();
            _optimizer.Step();

            if (_dqn.Model == _targetDqn.Model) return;
            if (_trainStep % 10000 == 0) Debug.Log($"Train Step: {_trainStep}");
            if (_trainStep % targetUpdateInterval != 0) return;
            _updateTargetOp.Dispatch();
        }

        public int TakeAction(float[] state)
        {
            _targetDqn.RuntimeInput.SetData(state);
            _targetDqn.RuntimeGraph.Forward();
            return SelectAction(_targetDqn.RuntimeGraph.Output, true);
        }

        public void Dispose()
        {
            _actionTensor.Dispose();
            foreach (var o in _targetQOp)
            {
                o.Dispose();
            }
            _targetQOp.Clear();
            _updateTargetOp.Dispose();
            _rewardTensor.Dispose();
            _doneTensor.Dispose();
            _dqn.Dispose();
            _targetDqn.Dispose();
            _loss.Dispose();
            _optimizer.Dispose();
            _clipGradNormOp.Dispose();
        }

        private int SelectAction(Tensor actionOutput, bool deterministic)
        {
            float explorationRate = -1;
            if (!deterministic)
            {
                if (_trainStep < _learningStarts)
                {
                    explorationRate = 1;
                }
                else
                {
                    var trainProgress = (float)_trainStep / _totalTrainSteps;
                    if (trainProgress > 0.1f) explorationRate = 0.05f;
                    else explorationRate = 1.0f - trainProgress * (1.0f - 0.05f) / 0.1f;
                }
            }

            if (explorationRate >= 0 && _rand.NextDouble() < explorationRate)
            {
                return _rand.Next(0, actionOutput.FlattenSize);
            }
            actionOutput.Max(out var maxIndex);
            return maxIndex;
        }
    }
}