using Blue.Data;
using Blue.Graph;
using Blue.Kit;
using Blue.Util;
using UnityEngine;

namespace Blue.Samples
{
    public class MnistSample : BaseSample
    {

        private readonly MnistData _data = new MnistData();

        private int _trainCorrectCount;
        private int _testCorrectCount;

        public override string Info
        {
            get
            {
                if (TrainCount <= 0) return "";
                var info = $"BatchTrainingTime: {BatchTrainingTime}";
                if (TestCount <= 0)
                {
                    info += $"\nEpoch: {Epoch}";
                    info += $"\nTraining: {TrainCount}/{_data.TrainData.Count}";
                    info += $"\nAccuracy: {_trainCorrectCount * 100 / TrainCount}%";
                }
                else
                {
                    info += $"\nTest: {TestCount}";
                    info += $"\nAccuracy: {_testCorrectCount * 100 / TestCount}%";
                }

                return info;
            }
        }

        protected override void OnEpochStart()
        {
            base.OnEpochStart();
            _trainCorrectCount = 0;
        }

        protected override void OnTrain(ComputeBuffer output, ComputeBuffer target)
        {
            if (output.MaxIndex() == target.MaxIndex()) _trainCorrectCount++;
        }

        protected override void OnTest(ComputeBuffer output, ComputeBuffer target)
        {
            if (output.MaxIndex() == target.MaxIndex()) _testCorrectCount++;
        }

        protected override int GetTrainCount()
        {
            return _data.TrainData.Count;
        }

        protected override void GetTrainData(int index, out float[] input, out float[] output)
        {
            input = _data.TrainData[index].ImageData;
            output = _data.TrainData[index].LabelArray;
        }

        protected override int GetTestCount()
        {
            return _data.TestData.Count;
        }

        protected override void GetTestData(int index, out float[] input, out float[] output)
        {
            input = _data.TestData[index].ImageData;
            output = _data.TestData[index].LabelArray;
        }

        protected override void SetupGraph(out IGraphNode input, out IGraphNode output)
        {
            _data.Load(Application.dataPath + "/Blue/Demo/MNist/train-labels-idx1-ubyte.bytes"
                , Application.dataPath + "/Blue/Demo/MNist/train-images-idx3-ubyte.bytes"
                , Application.dataPath + "/Blue/Demo/MNist/t10k-labels-idx1-ubyte.bytes"
                , Application.dataPath + "/Blue/Demo/MNist/t10k-images-idx3-ubyte.bytes");
            input = new DataNode("input", 28 * 28, false);
            var hidden = Layer.DenseLayer("hidden", input, 128, "relu");
            output = Layer.DenseLayer("output", hidden, 10, null);
        }
    }
}