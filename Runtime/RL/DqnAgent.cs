using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.NN;
using UnityEngine;

namespace Blue.RL
{
    public class DqnAgent : System.IDisposable
    {
        
        public interface IEnv
        {
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

            public ReplayMemory(int capacity, int stateSize, int actionSize)
            {
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
                    var i = Random.Range(0, _stateReplay.Count);
                    _sampleState.Add(_stateReplay[i]);
                    _sampleAction.Add(_actionReplay[i]);
                    _sampleReward.Add(_rewardReplay[i]);
                    _sampleNextState.Add(_nextStateReplay[i]);
                    _sampleDone.Add(_doneReplay[i]);
                }
            }
        }

        private readonly Dqn _dqn;
        private readonly Dqn _targetDqn;
        private readonly MseLoss _loss;
        private readonly IOptimizer _optimizer;
        private readonly Tensor _rewardTensor;
        private readonly Tensor _doneTensor;
        private readonly Tensor _actionTensor;
        private readonly List<Operate> _targetQOp = new List<Operate>();
        private readonly OperateList _updateTargetOp = new OperateList();
        private readonly ReplayMemory _memory;
        private readonly int _batchSize;

        private uint _trainStep;

        public DqnAgent(int stateSize
            , int actionSize
            , int batchSize
            , int replayBufferSize
            , Module qNetwork
            , Module targetQNetwork)
        {
            _batchSize = batchSize;
            _memory = new ReplayMemory(replayBufferSize, stateSize, actionSize);

            _rewardTensor = new Tensor(new []{batchSize, actionSize});
            _doneTensor = new Tensor(new []{batchSize, actionSize});
            _actionTensor = new Tensor(new []{batchSize, actionSize});

            _dqn = new Dqn(stateSize, batchSize, qNetwork);
            _targetDqn = new Dqn(stateSize, batchSize, targetQNetwork);
            
            _loss = new MseLoss(_dqn.TrainGraph.Output, _targetDqn.TrainGraph.Output);
            _optimizer = new AdamOptimizer(qNetwork.GetAllParameters());
            _targetQOp.Add(new Operate("Common/Translate", "CSMain")
                .SetTensor("weight", _doneTensor)
                .SetTensor("bias", _rewardTensor)
                .SetTensor("buffer", _targetDqn.TrainGraph.Output)
                .SetDispatchSize(_targetDqn.TrainGraph.Output.FlattenSize));
            _targetQOp.Add(Op.Lerp(_targetDqn.TrainGraph.Output, _dqn.TrainGraph.Output, _actionTensor));
            _targetDqn.Model.CopyParameter(_dqn.Model, _updateTargetOp);
        }

        public void Train(IEnv trainEnv
            , float greed = 0.2f
            , float futureRewardDiscount = 0.5f
            , uint targetUpdateInterval = 16)
        {
            _trainStep++;
            _memory.Push(out var state, out var action, out var reward, out var nextState, out var done);
            trainEnv.GetState(state);
            _dqn.RuntimeInput.SetData(state);
            _dqn.RuntimeGraph.Forward();
            var a = SelectAction(_dqn.RuntimeGraph.Output, greed);
            var d = trainEnv.Update(a, out var r);
            if (!d) trainEnv.GetState(nextState);
            for (var j = 0; j < reward.Length; j++)
            {
                reward[j] = r;
                done[j] = d ? 0f : futureRewardDiscount;
                action[j] = j == a ? 0 : 1;
            }
            if (_memory.Size < _batchSize) return;
                
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
            _optimizer.Step();

            if (_dqn.Model == _targetDqn.Model) return;
            if (_trainStep % targetUpdateInterval == 0) return;
            _updateTargetOp.Dispatch();
        }

        public int TakeAction(float[] state)
        {
            _targetDqn.RuntimeInput.SetData(state);
            _targetDqn.RuntimeGraph.Forward();
            return SelectAction(_targetDqn.RuntimeGraph.Output, 0f);
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
        }

        private static int SelectAction(Tensor actionOutput, float noise)
        {
            if (Random.Range(0f, 1f) < noise) return Random.Range(0, actionOutput.FlattenSize);
            actionOutput.Max(out var maxIndex);
            return maxIndex;
        }
    }
}