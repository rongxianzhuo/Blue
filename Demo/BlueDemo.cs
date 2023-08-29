using System.Collections;
using Blue.Core;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;
using UnityEngine.UI;

namespace Blue.Demo
{

    public class BlueDemo : MonoBehaviour
    {

        private readonly MnistData _mnistData = new MnistData();
        private readonly float[] _tempArray = new float[10];

        private Model _model;
        private TensorNode _input;
        private ComputeBuffer _target;
        private int _trainCount;
        private int _correctCount;
        [SerializeField] public Text infoText;

        private void Awake()
        {
            _target = new ComputeBuffer(10, 4);
            _input = new TensorNode("Input", 28 * 28, false);
            var hidden = Layer.DenseLayer("Hidden", _input, 128, "relu");
            var output = Layer.DenseLayer("Output", hidden, 10, "");
            _model = new Model(output);
            _model.EnableTrain(new AdamOptimizer(), "CrossEntropyLoss");
        }

        private void Start()
        {
            StartCoroutine(Train());
        }

        private void OnDestroy()
        {
            _target.Release();
            _model.Destroy();
        }

        private IEnumerator Train()
        {
            yield return _mnistData.DownloadData();
            var epoch = 0;
            while (true)
            {
                epoch++;
                _trainCount = 0;
                _correctCount = 0;
                for (var i = 0; i < _mnistData.TrainData.Count; i++)
                {
                    var data = _mnistData.TrainData[i];
                    _input.GetOutput().SetData(data.ImageData);
                    _target.SetData(data.LabelArray);
                    _model.Forward();
                    _model.Backward(_target);
                    _trainCount++;
                    if (OutputMaxIndex(_model.Output.GetOutput()) == OutputMaxIndex(_target)) _correctCount++;
                    if (i % 32 == 0)
                    {
                        infoText.text = $"Epoch: {epoch}\nTrainCount: {_trainCount}\nAccuracy: {_correctCount * 1000 / _trainCount}‰";
                        yield return null;
                    }
                }
            }
        }

        private int OutputMaxIndex(ComputeBuffer buffer)
        {
            buffer.GetData(_tempArray);
            var max = _tempArray[0];
            var index = 0;
            for (var i = 1; i < _tempArray.Length; i++)
            {
                if (_tempArray[i] <= max) continue;
                max = _tempArray[i];
                index = i;
            }
            return index;
        }
    }

}