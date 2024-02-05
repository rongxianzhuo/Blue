using System;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Data;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.Demo
{
    public class DqnTest : MonoBehaviour
    {

        public class Env
        {
            
            private const float PlayerStep = 0.02f;
            private const float BallStep = 0.01f;
            
            private static readonly float SqrtP5 = Mathf.Sqrt(0.5f);
            private float _distance;

            public Vector2 Player { get; private set; }

            public Vector2 Ball { get; private set; }
            
            private Vector2 RandomBornPosition =>
                new Vector2(Random.Range(-SqrtP5, SqrtP5), Random.Range(-SqrtP5, SqrtP5));

            public Env()
            {
                Reset();
            }
            
            public void Reset()
            {
                Player = RandomBornPosition;
                Ball = RandomBornPosition;
                while (Vector2.Distance(Player, Ball) < 0.5f) Reset();
                _distance = Vector2.Distance(Player, Ball);
            }

            public float[] GetState()
            {
                var state = new float[4];
                state[0] = Player.x;
                state[1] = Player.y;
                state[2] = Ball.x;
                state[3] = Ball.y;
                return state;
            }

            public float Update(int action)
            {
                var d = (Player - Ball).normalized;
                Ball += d * BallStep;
                Player += action switch
                {
                    1 => Vector2.up * PlayerStep,
                    2 => Vector2.right * PlayerStep,
                    3 => Vector2.down * PlayerStep,
                    4 => Vector2.left * PlayerStep,
                    _ => Vector2.zero
                };
                if (Vector2.Distance(Player, Ball) < 0.05f)
                {
                    Reset();
                    return -100;
                }

                if (Player.magnitude > 1)
                {
                    Reset();
                    return -100;
                }

                var distance = Vector2.Distance(Player, Ball);
                var reward = distance - _distance;
                _distance = distance;
                return reward;
            }

            public void Render(Transform player, Transform ball)
            {
                player.position = Player;
                ball.position = Ball;
            }
            
        }

        private const int BatchSize = 32;
        private const int ReplayBufferSize = 4;
        private const int ActionSize = 5;

        private readonly List<float[]> _inputReplay = new List<float[]>();
        private readonly List<float[]> _targetReplay = new List<float[]>();

        public Transform player;
        public Transform ball;

        private Env _runtimeEnv;
        private readonly Env[] _env = new Env[BatchSize];
        private Model _runtimeDqn;
        private Model _trainDqn;
        private ComputationalNode _runtimeInput;
        private ComputationalNode _trainInput;
        private Operate _loss;
        private Tensor _target;
        private IOptimizer _optimizer;
        private DatasetLoader _datasetLoader;
        
        private static string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "DqnSavedModel");

        private void Awake()
        {
            _runtimeEnv = new Env();
            for (var i = 0; i < _env.Length; i++)
            {
                _env[i] = new Env();
            }
            
            _runtimeInput = new ComputationalNode(false, 1, 4);
            var runtimeOutput = _runtimeInput.Linear(128).Activation("relu").Linear(ActionSize);
            _runtimeDqn = new Model(runtimeOutput, _runtimeInput);
            if (Directory.Exists(ModelSavePath)) _runtimeDqn.LoadParameterFile(ModelSavePath);
            
            _trainInput = new ComputationalNode(false, BatchSize, 4);
            var trainOutput = _trainInput.Linear(128).Activation("relu").Linear(ActionSize);
            _trainDqn = new Model(trainOutput, _trainInput);
            if (Directory.Exists(ModelSavePath)) _trainDqn.LoadParameterFile(ModelSavePath);
            _target = new Tensor(BatchSize, ActionSize);
            _loss = Op.L2Loss(_trainDqn.Output, _target, _trainDqn.Output.Gradient);
            _optimizer = new AdamOptimizer(_trainDqn.ParameterNodes);
            _datasetLoader = new DatasetLoader(BatchSize, BatchSize * ReplayBufferSize);
        }

        private void Update()
        {
            for (var i = 0; i < ReplayBufferSize; i++)
            {
                foreach (var env in _env)
                {
                    var state = env.GetState();
                    var qArray = new float[ActionSize];
                    _inputReplay.Add(state);
                    _targetReplay.Add(qArray);
                
                    _runtimeInput.SetData(state);
                    _runtimeDqn.Forward();
                    _runtimeDqn.Output.GetData(qArray);
                    var action = _runtimeDqn.Output.MaxIndex;
                    if (Random.Range(0f, 1f) < 0.1f) action = Random.Range(0, ActionSize);
                    var reward = env.Update(action);
                    if (reward < -99)
                    {
                        qArray[action] = reward;
                    }
                    else
                    {
                        var nextState = env.GetState();
                        _runtimeInput.SetData(nextState);
                        _runtimeDqn.Forward();
                        var targetQ = reward + 0.9f * _runtimeDqn.Output.Max;
                        qArray[action] = targetQ;
                    }
                }
            }

            {
                _runtimeInput.SetData(_runtimeEnv.GetState());
                _runtimeDqn.Forward();
                var action = _runtimeDqn.Output.MaxIndex;
                _runtimeEnv.Update(action);
                _runtimeEnv.Render(player, ball);
            }
            Shuffle();
            
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
                _trainDqn.CopyParameterTo(_runtimeDqn);
            }
            
            _inputReplay.Clear();
            _targetReplay.Clear();
        }

        private void OnDestroy()
        {
            _trainDqn.SaveParameterFile(ModelSavePath);
            _runtimeDqn.Dispose();
            _runtimeInput.Dispose();
            _trainDqn.Dispose();
            _trainInput.Dispose();
            _target.Dispose();
            _loss.Dispose();
            _optimizer.Dispose();
            _datasetLoader.Dispose();
        }

        private void Shuffle()
        {
            for (var i = 0; i < _inputReplay.Count; i++)
            {
                var j = Random.Range(0, _inputReplay.Count);
                var k = Random.Range(0, _inputReplay.Count);
                (_inputReplay[j], _inputReplay[k]) = (_inputReplay[k], _inputReplay[j]);
                (_targetReplay[j], _targetReplay[k]) = (_targetReplay[k], _targetReplay[j]);
            }
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(default, 1f);
        }
    }
}