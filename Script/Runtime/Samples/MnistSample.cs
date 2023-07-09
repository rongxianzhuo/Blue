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

        private int _correctCount;

        public override string Info
        {
            get
            {
                if (TestCount <= 0) return $"Training: {TrainCount}/{_data.TrainData.Count}";
                return $"Accuracy: {_correctCount * 100 / TestCount}% ({TestCount})";
            }
        }

        protected override void OnTest(ComputeBuffer output, ComputeBuffer target)
        {
            if (output.MaxIndex() == target.MaxIndex()) _correctCount++;
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
            var hidden1 = Layer.DenseLayer(input, 128, "relu");
            var hidden2 = Layer.DenseLayer(hidden1, 10, null);
            output = new SoftmaxNode(hidden2);
        }
    }
}