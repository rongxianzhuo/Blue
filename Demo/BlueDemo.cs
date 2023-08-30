using System.Collections;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;
using UnityEngine.UI;

namespace Blue.Demo
{

    public class BlueDemo : MonoBehaviour
    {
        
        public Text infoText;

        private SimpleModel _model;

        private void Awake()
        {
            var input = new TensorNode("Input", 28 * 28, false);
            var hidden = Layer.DenseLayer("Hidden", input, 128, "relu");
            var output = Layer.DenseLayer("Output", hidden, 10);
            _model = new SimpleModel(input, output);
            _model.EnableTrain(new AdamOptimizer(), "CrossEntropyLoss");
            StartCoroutine(Train());
        }

        private void OnDestroy()
        {
            _model.Destroy();
        }

        private IEnumerator Train()
        {
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            var epoch = 0;
            while (epoch < 5)
            {
                epoch++;
                var correctCount = 0;
                for (var i = 0; i < mnistData.TrainData.Count; i++)
                {
                    var data = mnistData.TrainData[i];
                    _model.Train(data.ImageData, data.LabelArray);
                    if (_model.GetMaxOutputIndex() == data.Label) correctCount++;
                    if (i % 32 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nTrainCount: {i + 1}\nAccuracy: {correctCount * 1000 / (i + 1)}‰";
                    yield return null;
                }
            }
        }
    }

}