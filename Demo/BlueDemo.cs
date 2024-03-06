using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Data;
using Blue.Graph;
using Blue.Kit;
using Blue.Optimizers;
using Blue.Runtime.NN;
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
            using var model = new Sequential(new Linear(784, 128), new Activation("relu"), new Linear(128, 10));
            if (File.Exists(ModelSavePath)) model.LoadFromFile(ModelSavePath);
            var trainInput = new ComputationalNode(false, BatchSize, 784);
            using var trainGraph = new ComputationalGraph(model.Forward(trainInput));
            using var target = new Tensor(BatchSize, 10);
            using var crossEntropyLoss = Op.CrossEntropyLoss(trainGraph.Output, target, trainGraph.Output.Gradient);
            using var optimizer = new AdamOptimizer(model.GetAllParameters());
            
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
                    trainGraph.Output.Backward();
                    optimizer.Step();
                    if (i % 128 != 0) continue;
                    infoText.text = $"Epoch: {epoch}\nStep: {i + 1}/{datasetLoader.BatchCount}";
                    yield return null;
                }
            }
            
            if (saveModel) model.SaveToFile(ModelSavePath);
            
            // evaluate
            var sampleCount = mnistData.TestInputData.Count;
            var input = new ComputationalNode(false, sampleCount, 784);
            using var testGraph = new ComputationalGraph(model.Forward(input));
            input.SetData(mnistData.TestInputData);
            testGraph.Forward();
            infoText.text = $"Accuracy: {GetCorrectCount(testGraph.Output, mnistData.TestOutputLabel) * 100f / sampleCount:0.00}%";
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