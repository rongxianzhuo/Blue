using System.IO;
using Blue.Kit;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.Demo
{
    public class DqnTest : MonoBehaviour
    {

        public class Env : DqnBrain.IEnv
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
            
            private void Reset()
            {
                Player = RandomBornPosition;
                Ball = RandomBornPosition;
                while (Vector2.Distance(Player, Ball) < 0.5f) Reset();
                _distance = Vector2.Distance(Player, Ball);
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
                if (Vector2.Distance(Player, Ball) < 0.05f)
                {
                    Reset();
                    reward = -100;
                    return false;
                }

                if (Player.magnitude > 1)
                {
                    Reset();
                    reward = -100;
                    return false;
                }

                var distance = Vector2.Distance(Player, Ball);
                reward = distance - _distance;
                _distance = distance;
                return true;
            }

            public void Render()
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(Player, 0.025f);
                Gizmos.DrawSphere(Ball, 0.025f);
            }
            
        }

        private Env _previewEnv;
        private Env _trainEnv;
        private DqnBrain _dqnBrain;
        
        private static string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "DqnSavedModel");

        private void Awake()
        {
            _previewEnv = new Env();
            _trainEnv = new Env();
            _dqnBrain = new DqnBrain(4, 5);
        }

        private void Update()
        {
            _dqnBrain.TrainUpdate(_trainEnv, _previewEnv);
        }

        private void OnDestroy()
        {
            _dqnBrain.Dispose();
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(default, 1f);
            _previewEnv?.Render();
        }
    }
}