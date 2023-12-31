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

        private static string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "SavedModel");

        private void Awake()
        {
            _target = new Tensor(BatchSize, 10);
            _input = new ComputationalNode(false, BatchSize, 784);
            _model = new Model(_input.Linear(128).Activation("relu").Linear(10));
            _optimizer = new AdamOptimizer();
            _crossEntropyLoss = Op.CrossEntropyLoss(_model.Output.Output, _target, _model.Output.Gradient);
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
            // download mnist data
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            
            // init dataset loader
            var x = new List<float[]>();
            var y = new List<float[]>();
            foreach (var data in mnistData.TrainData)
            {
                x.Add(data.ImageData);
                y.Add(data.LabelArray);
            }
            _datasetLoader = new DatasetLoader(BatchSize, mnistData.TrainData.Count);
            _datasetLoader.LoadDataset(x, _input.Output);
            _datasetLoader.LoadDataset(y, _target);
            
            // train model
            var epoch = 0;
            while (epoch++ < trainEpochs)
            {
                for (var i = 0; i < _datasetLoader.BatchCount; i++)
                {
                    _datasetLoader.LoadBatch(i);
                    _model.Forward();
                    _model.ClearGradient();
                    _crossEntropyLoss.Dispatch();
                    _model.Backward();
                    _optimizer.Step(_model.ParameterNodes);
                    if (i % 64 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{_datasetLoader.BatchCount}";
                    yield return null;
                }
            }

            // save model
            if (trainEpochs > 0)
            {
                _model.SaveParameterFile(ModelSavePath);
                Debug.Log("Model saved");
            }
            
            // evaluate
            Evaluate(mnistData);
        }

        private void Evaluate(MnistData mnistData)
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
            input.Output.SetData(x);
            model.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(model, y) * 100f / sampleCount:0.00}%";
            model.Destroy();
            input.Destroy();
            return;
            static int GetCorrectCount(Model model, IReadOnlyList<int> batchTargetLabel)
            {
                var correctCount = 0;
                var outputData = model.Output.Output.Sync();
                for (var i = 0; i < batchTargetLabel.Count; i++)
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

}