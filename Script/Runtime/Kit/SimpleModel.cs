using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Kit
{
    public class SimpleModel : Model
    {

        private readonly TensorNode _input;
        private readonly Tensor _target;
        private readonly float[] _outputArray;

        private Tensor _trainX;
        private Tensor _trainY;
        
        public SimpleModel(TensorNode input, IGraphNode outputNode) : base(outputNode)
        {
            _input = input;
            _target = new Tensor(outputNode.GetOutput().Size);
            _outputArray = new float[outputNode.GetOutput().Size];
        }

        public void StartTrain(List<float> x, List<float> y)
        {
            StopTrain();
            _trainX = new Tensor(x);
            _trainY = new Tensor(y);
        }

        public void UpdateTrain(int sampleIndex)
        {
            var inputLength = _input.GetOutput().Size;
            var outputLength = _target.Size;
            Op.Copy(_trainX, sampleIndex * inputLength, _input.GetOutput(), 0, inputLength);
            Op.Copy(_trainY, sampleIndex * outputLength, _target, 0, outputLength);
            Forward();
            Backward(_target);
        }

        public void StopTrain()
        {
            _trainX?.Release();
            _trainY?.Release();
            _trainX = null;
            _trainY = null;
        }

        public override void Destroy()
        {
            StopTrain();
            _target.Release();
            base.Destroy();
        }

        public int GetMaxOutputIndex()
        {
            Output.GetOutput().GetData(_outputArray);
            var max = _outputArray[0];
            var index = 0;
            for (var i = 1; i < _outputArray.Length; i++)
            {
                if (_outputArray[i] <= max) continue;
                max = _outputArray[i];
                index = i;
            }
            return index;
        }
    }
}