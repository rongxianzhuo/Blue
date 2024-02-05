using System.Collections;
using System.Collections.Generic;
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

        private void Awake()
        {
            StartCoroutine(Run());
        }

        private IEnumerator Run()
        {
            // download mnist data
            var mnistData = new MnistData();
            yield return mnistData.DownloadData();
            
            // create model
            using var trainInput = new ComputationalNode(false, BatchSize, 784);
            using var trainModel = new Model(trainInput.Linear(128).Activation("relu").Linear(10));
            using var target = new Tensor(BatchSize, 10);
            using var crossEntropyLoss = Op.CrossEntropyLoss(trainModel.Output, target, trainModel.Output.Gradient);
            using var optimizer = new AdamOptimizer(trainModel.ParameterNodes);
            
            // init dataset loader
            using var datasetLoader = new DatasetLoader(BatchSize, mnistData.TrainInputData.Count);
            datasetLoader.LoadDataset(mnistData.TrainInputData, trainInput);
            datasetLoader.LoadDataset(mnistData.TrainOutputData, target);
            
            // train model
            var epoch = 0;
            while (epoch++ < trainEpochs)
            {
                for (var i = 0; i < datasetLoader.BatchCount; i++)
                {
                    datasetLoader.LoadBatch(i);
                    trainModel.Forward();
                    trainModel.ClearGradient();
                    crossEntropyLoss.Dispatch();
                    trainModel.Backward();
                    optimizer.Step();
                    if (i % 64 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{datasetLoader.BatchCount}";
                    yield return null;
                }
            }
            
            // evaluate
            var sampleCount = mnistData.TestInputData.Count;
            using var input = new ComputationalNode(false, sampleCount, 784);
            using var model = new Model(input.Linear(128).Activation("relu").Linear(10));
            trainModel.CopyParameterTo(model);
            input.SetData(mnistData.TestInputData);
            model.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(model, mnistData.TestOutputLabel) * 100f / sampleCount:0.00}%";
        }
        
        private static int GetCorrectCount(Model model, IReadOnlyList<int> batchTargetLabel)
        {
            var correctCount = 0;
            var outputData = model.Output.Sync();
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