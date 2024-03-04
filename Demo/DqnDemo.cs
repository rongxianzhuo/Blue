using System.Collections.Generic;
using Blue.Core;
using Blue.RL;
using Blue.Runtime.NN;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.Demo
{
    public class DqnDemo : MonoBehaviour
    {

        public class Env : DqnBrain.IEnv
        {
            
            private const float PlayerStep = 0.02f;
            private const float BallStep = 0.01f;
            
            private static readonly float SqrtP5 = Mathf.Sqrt(0.5f);

            private int _lastAction = -1;

            public Vector2 Player { get; private set; }

            public Vector2 Ball { get; private set; }
            
            private Vector2 RandomBornPosition =>
                new Vector2(Random.Range(-SqrtP5, SqrtP5), Random.Range(-SqrtP5, SqrtP5));

            public Env()
            {
                Reset();
            }
            
            private void Reset()
            {
                Player = RandomBornPosition;
                Ball = RandomBornPosition;
                while (Vector2.Distance(Player, Ball) < 0.5f) Reset();
            }

            public void GetState(float[] state)
            {
                state[0] = Player.x;
                state[1] = Player.y;
                state[2] = Ball.x;
                state[3] = Ball.y;
            }

            public bool Update(int action, out float reward)
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
                if (_lastAction != -1 && _lastAction != action) reward = -0.1f;
                else reward = 1;
                _lastAction = action;
                if (Vector2.Distance(Player, Ball) < 0.05f)
                {
                    Reset();
                    reward = -10;
                    return true;
                }

                if (Player.magnitude > 1)
                {
                    Reset();
                    reward = -10;
                    return true;
                }

                return false;
            }

            public void Render()
            {
                Gizmos.color = Color.green;
                Gizmos.DrawSphere(Player, 0.025f);
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(Ball, 0.025f);
            }
            
        }

        private Env _previewEnv;
        private Env _trainEnv;
        private DqnBrain _dqn;
        private Module _qNetwork;
        private Module _targetQNetwork;
        private float[] _tempState;
        private float _greed = 0.5f;

        private void Awake()
        {
            _tempState = new float[4];
            _previewEnv = new Env();
            _trainEnv = new Env();
            _qNetwork = new Sequential(new Linear(4, 128)
                , new Activation("relu")
                , new Linear(64, 64)
                , new Activation("relu")
                , new Linear(64, 5));
            _targetQNetwork = new Sequential(new Linear(4, 128)
                , new Activation("relu")
                , new Linear(64, 64)
                , new Activation("relu")
                , new Linear(64, 5));
            var list = new List<Operate>();
            _targetQNetwork.CopyParameter(list, _qNetwork);
            foreach (var o in list)
            {
                o.Dispatch();
                o.Dispose();
            }
            list.Clear();
            _dqn = new DqnBrain(4
                , 5
                , 32
                , 10000
                , _qNetwork
                , _targetQNetwork);
        }

        private void Update()
        {
            for (var i = 0; i < 32; i++)
            {
                _dqn.Train(_trainEnv, greed: _greed);
                _greed = Mathf.Max(0.01f, _greed - 0.00001f);
            }
            _previewEnv.GetState(_tempState);
            _previewEnv.Update(_dqn.TakeAction(_tempState), out _);
        }

        private void OnDestroy()
        {
            _dqn.Dispose();
            _qNetwork.Dispose();
            _targetQNetwork.Dispose();
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(default, 1f);
            _previewEnv?.Render();
        }
    }
}