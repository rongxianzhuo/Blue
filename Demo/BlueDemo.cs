using System;
using System.Collections;
using System.IO;
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

        private const int BatchSize = 32;

        public bool saveModel;
        public Text infoText;
        public int trainEpochs = 5;

        private Model _model;
        private TensorNode _input;
        private Tensor _target;

        private string ModelSavePath => $"{Application.dataPath}/Blue/Demo/SavedModel";

        private void Awake()
        {
            _target = new Tensor(BatchSize, 10);
            _model = new ModelBuilder()
                .Tensor(false, out _input, BatchSize, 784)
                .Linear(128)
                .Activation("relu")
                .Linear(10)
                .Build();
            _model.EnableTrain(new AdamOptimizer(), "CrossEntropyLoss");
            if (Directory.Exists(ModelSavePath)) _model.LoadParameterFile(ModelSavePath);
            StartCoroutine(Train());
        }

        private void OnDestroy()
        {
            _model.Destroy();
            _target.Release();
        }

        private IEnumerator Train()
        {
            var batchTargetLabel = new int[BatchSize];
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            var x = new float[BatchSize * 784];
            var y = new float[BatchSize * 10];
            var epoch = 0;
            while (epoch < trainEpochs)
            {
                epoch++;
                var batchCount = mnistData.TrainData.Count / BatchSize;
                for (var i = 0; i < batchCount; i++)
                {
                    for (var j = 0; j < BatchSize; j++)
                    {
                        Array.Copy(mnistData.TrainData[i * BatchSize + j].ImageData, 0, x, j * 784, 784);
                        Array.Copy(mnistData.TrainData[i * BatchSize + j].LabelArray, 0, y, j * 10, 10);
                        batchTargetLabel[j] = mnistData.TrainData[i * BatchSize + j].Label;
                    }
                    _input.GetOutput().SetData(x);
                    _target.SetData(y);
                    _model.Forward();
                    _model.Backward(_target);
                    if (i % 32 == 0)
                    {
                        infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{batchCount}";
                        yield return null;
                    }
                }
                if (saveModel)
                {
                    _model.SaveParameterFile(ModelSavePath);
                    Debug.Log("Model saved");
                }
            }
            Test(mnistData);
        }

        private void Test(MnistData mnistData)
        {
            const int sampleCount = 1024;
            _input.Resize(sampleCount, 784);
            var x = new float[sampleCount * 784];
            var y = new int[sampleCount];
            for (var i = 0; i < sampleCount; i++)
            {
                Array.Copy(mnistData.TestData[i].ImageData, 0, x, i * 784, 784);
                y[i] = mnistData.TestData[i].Label;
            }
            _input.GetOutput().SetData(x);
            _model.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(y) * 100f / sampleCount:0.00}%";
        }

        public int GetCorrectCount(int[] batchTargetLabel)
        {
            var correctCount = 0;
            var outputData = _model.Output.GetOutput().Sync();
            for (var i = 0; i < batchTargetLabel.Length; i++)
            {
                var max = outputData[i * 10];
                var index = 0;
                for (var j = 1; j < 10; j++)
                {
                    if (outputData[i * 10 + j] <= max) continue;
                    max = outputData[i * 10 + j];
                    index = j;
                }

                if (index == batchTargetLabel[i]) correctCount++;
            }
            return correctCount;
        }
    }

}