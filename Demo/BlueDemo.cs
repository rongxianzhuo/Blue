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

        public bool saveModel;
        public Text infoText;
        public int trainEpochs = 2;
        
        private static string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "MnistSavedModel");

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
            var trainInput = new ComputationalNode(false, BatchSize, 784);
            var output = trainInput.Linear(128).Activation("relu").Linear(10);
            var trainGraph = new ComputationalGraph(output);
            using var target = new Tensor(BatchSize, 10);
            using var crossEntropyLoss = Op.CrossEntropyLoss(output, target, output.Gradient);
            using var optimizer = new AdamOptimizer(trainGraph.ParameterNodes);
            
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
                    trainGraph.Forward();
                    trainGraph.ClearGradient();
                    crossEntropyLoss.Dispatch();
                    trainGraph.Backward();
                    optimizer.Step();
                    if (i % 16 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{datasetLoader.BatchCount}";
                    yield return null;
                }
            }
            
            if (saveModel) trainGraph.SaveParameterFile(ModelSavePath);
            
            // evaluate
            var sampleCount = mnistData.TestInputData.Count;
            var input = new ComputationalNode(false, sampleCount, 784);
            var testOutput = input.Linear(128).Activation("relu").Linear(10);
            var testGraph = new ComputationalGraph(testOutput);
            if (Directory.Exists(ModelSavePath)) testGraph.LoadParameterFile(ModelSavePath);
            else trainGraph.CopyParameterTo(testGraph);
            input.SetData(mnistData.TestInputData);
            testGraph.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(testOutput, mnistData.TestOutputLabel) * 100f / sampleCount:0.00}%";
            
            // Release asset
            trainGraph.DisposeNodes();
            testGraph.DisposeNodes();
        }
        
        private static int GetCorrectCount(ComputationalNode output, IReadOnlyList<int> batchTargetLabel)
        {
            var correctCount = 0;
            var outputData = output.Sync();
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