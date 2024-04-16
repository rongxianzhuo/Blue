using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Data;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.NN;
using UnityEngine;

namespace Blue.Demo
{

    public class MnistDemo : MonoBehaviour
    {

        private class Model : Module
        {

            private readonly Linear _fc1;
            private readonly Linear _fc2;

            public Model()
            {
                _fc1 = RegisterModule(new Linear(784, 128));
                _fc2 = RegisterModule(new Linear(128, 10));
            }
            
            public override ComputationalNode Build(params ComputationalNode[] input)
            {
                var x = input[0];
                x = _fc1.Build(x).ReLU();
                return _fc2.Build(x);
            }
        }

        private const int BatchSize = 32;

        public bool saveModel;
        public int trainEpochs = 2;
        
        private static string ModelSavePath => Path.Combine(Application.dataPath, "Blue", "Demo", "BlueDemoModel.bytes");

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
            using var model = new Model();
            if (File.Exists(ModelSavePath)) model.LoadFromFile(ModelSavePath);
            var trainInput = new ComputationalNode(false, BatchSize, 784);
            using var trainGraph = model.Build(trainInput).Graph();
            using var target = new Tensor(BatchSize, 10);
            using var crossEntropyLoss = Op.CrossEntropyLoss(trainGraph.Output, target, trainGraph.Output.Gradient);
            using var optimizer = new AdamOptimizer(model.GetAllParameters());
            
            // init dataset loader
            using var datasetLoader = new DatasetLoader(BatchSize, mnistData.TrainInputData.Count);
            datasetLoader.LoadDataset(mnistData.TrainInputData, trainInput);
            datasetLoader.LoadDataset(mnistData.TrainOutputData, target);
            
            // evaluate
            var sampleCount = mnistData.TestInputData.Count;
            var testInput = new ComputationalNode(false, sampleCount, 784);
            testInput.SetData(mnistData.TestInputData);
            using var testGraph = model.Build(testInput).Graph();
            
            // train model
            var epoch = 0;
            while (epoch < trainEpochs)
            {
                testGraph.Forward();
                Debug.Log($"Epoch: {epoch}, Accuracy: {GetCorrectCount(testGraph.Output, mnistData.TestOutputLabel) * 100f / sampleCount:0.00}%");
                for (var i = 0; i < datasetLoader.BatchCount; i++)
                {
                    datasetLoader.LoadBatch(i);
                    trainGraph.Forward();
                    trainGraph.ClearGradient();
                    crossEntropyLoss.Dispatch();
                    trainGraph.Output.Backward();
                    optimizer.Step();
                    if (i % 128 != 0) continue;
                    yield return null;
                }
                epoch++;
            }
            
            // test model
            testGraph.Forward();
            Debug.Log($"Epoch: {epoch}, Accuracy: {GetCorrectCount(testGraph.Output, mnistData.TestOutputLabel) * 100f / sampleCount:0.00}%");
            
            if (saveModel) model.SaveToFile(ModelSavePath);
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