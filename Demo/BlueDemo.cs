using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.Util;
using UnityEngine;
using UnityEngine.UI;
using Model = Blue.Kit.Model;
using Random = UnityEngine.Random;

namespace Blue.Samples
{

    public class BlueDemo : MonoBehaviour
    {

        public const int Segment = 64;
        public const float Min = -10f;
        public const float Max = 10f;

        [Range(1, 32)]
        public int batchSize = 16;

        [Range(0.01f, 0.1f)]
        public float learningRate = 0.05f;

        [Range(1, 64)]
        public int hiddenLayerSize = 16;
        
        public Text lossText;

        private Model _model;
        private DataNode _input;
        private IGraphNode _output;
        private ComputeBuffer _target;
        private readonly float[] _inputArray = new float[Segment + 1];
        private readonly float[] _outputArray = new float[Segment + 1];
        private readonly float[] _tempArray = new float[1];

        private static float Equation(float x)
        {
            return Mathf.Sin(x) * 8;
        }

        private void Awake()
        {
            _target = new ComputeBuffer(1, 4);
            _model = new Model();
            _input = new DataNode(1, false);
            var layer0 = Layer.DenseLayer(_input, hiddenLayerSize, "sigmoid");
            _output = Layer.DenseLayer(layer0, 1, null);
            _model.Load(_output, new AdamOptimizer(learningRate), batchSize);
        }

        private void Update()
        {
            var loss = 0f;
            for (var i = 0; i < batchSize; i++)
            {
                var x = Random.Range(Min, Max);
                var y = Equation(x);
                _tempArray[0] = x;
                _input.GetOutput().SetData(_tempArray);
                _tempArray[0] = y;
                _target.SetData(_tempArray);
                
                _model.ForwardPropagation();
                LossFunction.L2Loss(_output, _target);
                loss += _output.GetGradient().SquareSum() / _output.GetGradient().count;
                _model.BackwardPropagation();
            }

            lossText.text = $"Loss: {loss / batchSize}";
        }

        private void OnDestroy()
        {
            _model.Unload();
            _target.Release();
        }
        
        private void OnDrawGizmos()
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(new Vector3(-10, -10, 0), new Vector3(-10, 10, 0));
            Gizmos.DrawLine(new Vector3(-10, -10, 0), new Vector3(10, -10, 0));
            Gizmos.DrawLine(new Vector3(10, 10, 0), new Vector3(-10, 10, 0));
            Gizmos.DrawLine(new Vector3(10, 10, 0), new Vector3(10, -10, 0));
            
            Gizmos.color = Color.green;
            var start = new Vector3(Min, Equation(Min), 0);
            for (var i = 1; i <= Segment; i++)
            {
                var x = Min + i * (Max - Min) / Segment;
                var next = new Vector3(x, Equation(x), 0);
                Gizmos.DrawLine(start, next);
                start = next;
            }

            if (_model == null) return;
            Gizmos.color = Color.red;
            for (var i = 0; i <= Segment; i++)
            {
                var x = Min + i * (Max - Min) / Segment;
                _inputArray[i] = x;
                _tempArray[0] = x;
                _input.GetOutput().SetData(_tempArray);
                _model.ForwardPropagation();
                _output.GetOutput().GetData(_tempArray);
                _outputArray[i] = _tempArray[0];
            }
            start = new Vector3(_inputArray[0], _outputArray[0], 0);
            for (var i = 1; i <= Segment; i++)
            {
                var x = _inputArray[i];
                var y = _outputArray[i];
                var next = new Vector3(x, y, 0);
                Gizmos.DrawLine(start, next);
                start = next;
            }
        }
    }

}