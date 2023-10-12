using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Data;
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

        public Text infoText;
        public int trainEpochs = 2;

        private Model _model;
        private ComputationalNode _input;
        private Tensor _target;
        private IOptimizer _optimizer;
        private Operate _crossEntropyLoss;
        private DatasetLoader _datasetLoader;

        private string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "SavedModel");

        private void Awake()
        {
            _target = new Tensor(BatchSize, 10);
            _input = new ComputationalNode(false, BatchSize, 784);
            _model = new Model(_input.Linear(128).Activation("relu").Dropout(0.2f).Linear(10));
            _optimizer = new AdamOptimizer();
            _crossEntropyLoss = Op.CrossEntropyLoss(_model.Output.GetOutput(), _target, _model.Output.GetGradient());
            if (Directory.Exists(ModelSavePath)) _model.LoadParameterFile(ModelSavePath);
            StartCoroutine(Train());
        }

        private void OnDestroy()
        {
            _input.Destroy();
            _model.Destroy();
            _target.Release();
            _optimizer.Destroy();
            _crossEntropyLoss.Destroy();
            _datasetLoader?.Destroy();
        }

        private IEnumerator Train()
        {
            var batchTargetLabel = new int[BatchSize];
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            var x = new List<float[]>();
            var y = new List<float[]>();
            foreach (var data in mnistData.TrainData)
            {
                x.Add(data.ImageData);
                y.Add(data.LabelArray);
            }
            _datasetLoader = new DatasetLoader(BatchSize, mnistData.TrainData.Count);
            _datasetLoader.LoadDataset(x, _input.GetOutput());
            _datasetLoader.LoadDataset(y, _target);
            var epoch = 0;
            while (epoch < trainEpochs)
            {
                epoch++;
                var batchCount = mnistData.TrainData.Count / BatchSize;
                for (var i = 0; i < batchCount; i++)
                {
                    for (var j = 0; j < BatchSize; j++)
                    {
                        batchTargetLabel[j] = mnistData.TrainData[i * BatchSize + j].Label;
                    }
                    _datasetLoader.LoadBatch(i);
                    _model.Forward();
                    _model.ClearGradient();
                    _crossEntropyLoss.Dispatch();
                    _model.Backward();
                    _optimizer.Step(_model.ParameterNodes);
                    if (i % 32 == 0)
                    {
                        infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{batchCount}";
                        yield return null;
                    }
                }
            }

            if (trainEpochs > 0)
            {
                _model.SaveParameterFile(ModelSavePath);
                Debug.Log("Model saved");
            }
            Test(mnistData);
        }

        private void Test(MnistData mnistData)
        {
            var sampleCount = mnistData.TestData.Count;
            var input = new ComputationalNode(false, sampleCount, 784);
            var model = new Model(input.Linear(128).Activation("relu").Linear(10));
            if (Directory.Exists(ModelSavePath)) model.LoadParameterFile(ModelSavePath);
            var x = new float[sampleCount * 784];
            var y = new int[sampleCount];
            for (var i = 0; i < sampleCount; i++)
            {
                Array.Copy(mnistData.TestData[i].ImageData, 0, x, i * 784, 784);
                y[i] = mnistData.TestData[i].Label;
            }
            input.GetOutput().SetData(x);
            model.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(model, y) * 100f / sampleCount:0.00}%";
            model.Destroy();
            input.Destroy();
        }

        public static int GetCorrectCount(Model model, int[] batchTargetLabel)
        {
            var correctCount = 0;
            var outputData = model.Output.GetOutput().Sync();
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