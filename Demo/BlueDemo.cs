using System;
using System.Collections;
using System.Collections.Generic;
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

        public bool loadModel;
        public bool saveModel;
        public Text infoText;
        public int trainEpochs = 2;

        private Model _model;
        private TensorNode _input;
        private Tensor _target;
        private IOptimizer _optimizer;

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
            _optimizer = new AdamOptimizer();
            if (loadModel && Directory.Exists(ModelSavePath)) _model.LoadParameterFile(ModelSavePath);
            StartCoroutine(Train());
        }

        private void OnDestroy()
        {
            _model.Destroy();
            _target.Release();
            _optimizer.Destroy();
        }

        private IEnumerator Train()
        {
            var batchTargetLabel = new int[BatchSize];
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            var xList = new List<float>();
            var yList = new List<float>();
            foreach (var data in mnistData.TrainData)
            {
                xList.AddRange(data.ImageData);
                yList.AddRange(data.LabelArray);
            }
            var xTensor = new Tensor(xList);
            var yTensor = new Tensor(yList);
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
                    Op.Copy(xTensor
                        , i * BatchSize * 784
                        , 0
                        , _input.GetOutput()
                        , 0
                        , 0
                        , BatchSize * 784
                        , BatchSize * 784);
                    Op.Copy(yTensor
                        , i * BatchSize * 10
                        , 0
                        , _target
                        , 0
                        , 0
                        , BatchSize * 10
                        , BatchSize * 10);
                    _model.Forward();
                    Op.CrossEntropyLoss(_model.Output.GetOutput(), _target, _model.Output.GetGradient());
                    _model.Backward();
                    _model.ForeachParameterNode(_optimizer.Step);
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
            xTensor.Release();
            yTensor.Release();
            Test(mnistData);
        }

        private void Test(MnistData mnistData)
        {
            const int sampleCount = 2048;
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