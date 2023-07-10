using System.Text;
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
                if (TestCount <= 0)
                {
                    return $"Epoch: {Epoch}\nTraining: {TrainCount}/{_data.TrainData.Count}\nAccuracy: {_trainCorrectCount * 100 / TrainCount}%";
                }
                return $"Test: {TestCount}\nAccuracy: {_testCorrectCount * 100 / TestCount}%";
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
            _data.Load(Application.dataPath + "/GameFramework/ML/Runtime/MNist/train-labels-idx1-ubyte.bytes"
                , Application.dataPath + "/GameFramework/ML/Runtime/MNist/train-images-idx3-ubyte.bytes"
                , Application.dataPath + "/GameFramework/ML/Runtime/MNist/t10k-labels-idx1-ubyte.bytes"
                , Application.dataPath + "/GameFramework/ML/Runtime/MNist/t10k-images-idx3-ubyte.bytes");
            input = new DataNode(28 * 28, false);
            var hidden = Layer.DenseLayer(input, 128, "relu");
            output = Layer.DenseLayer(hidden, 10, null);
        }
    }
}