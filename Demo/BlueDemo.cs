using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using UnityEngine;
using UnityEngine.UI;

namespace Blue.Demo
{

    public class BlueDemo : MonoBehaviour
    {

        public bool saveModel;
        public Text infoText;

        private SimpleModel _model;

        private string ModelSavePath => $"{Application.dataPath}/Blue/Demo/SavedModel";

        private void Awake()
        {
            _model = new ModelBuilder()
                .TensorNode("Input", 28 * 28, false)
                .DenseLayer("Hidden", "Input", 128, "relu")
                .DenseLayer("Output", "Hidden", 10)
                .BuildSimpleModel();
            _model.EnableTrain(new AdamOptimizer(), "CrossEntropyLoss");
            if (Directory.Exists(ModelSavePath)) _model.LoadParameterFile(ModelSavePath);
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
            var x = new List<float>(mnistData.TrainData.Count);
            var y = new List<float>(mnistData.TrainData.Count);
            foreach (var data in mnistData.TrainData)
            {
                x.AddRange(data.ImageData);
                y.AddRange(data.LabelArray);
            }
            _model.StartTrain(y, x);
            var epoch = 0;
            while (epoch < 5)
            {
                epoch++;
                var correctCount = 0;
                for (var i = 0; i < mnistData.TrainData.Count; i++)
                {
                    var data = mnistData.TrainData[i];
                    _model.UpdateTrain(i);
                    if (_model.GetMaxOutputIndex() == data.Label) correctCount++;
                    if (i % 32 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nTrainCount: {i + 1}\nAccuracy: {correctCount * 100f / (i + 1):0.00}%";
                    yield return null;
                }
                if (saveModel)
                {
                    _model.SaveParameterFile(ModelSavePath);
                    Debug.Log("Model saved");
                }
            }
        }
    }

}