using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Kit
{
    public class SimpleModel : Model
    {

        private readonly TensorNode[] _input;
        private readonly Tensor _target;
        private readonly float[] _outputArray;

        private Tensor[] _trainX;
        private Tensor _trainY;
        
        public SimpleModel(IGraphNode outputNode, params TensorNode[] input) : base(outputNode)
        {
            _input = input;
            _target = new Tensor(outputNode.GetOutput().Size);
            _outputArray = new float[outputNode.GetOutput().FlattenSize];
        }

        public void StartTrain(List<float> y, params List<float>[] x)
        {
            StopTrain();
            _trainX = new Tensor[x.Length];
            for (var i = 0; i < x.Length; i++)
            {
                _trainX[i] = new Tensor(x[i]);
            }
            _trainY = new Tensor(y);
        }

        public void UpdateTrain(int sampleIndex)
        {
            var outputLength = _target.FlattenSize;
            for (var i = 0; i < _input.Length; i++)
            {
                var inputLength = _input[i].GetOutput().FlattenSize;
                Op.Copy(_trainX[i], sampleIndex * inputLength, _input[i].GetOutput(), 0, inputLength);
            }
            Op.Copy(_trainY, sampleIndex * outputLength, _target, 0, outputLength);
            Forward();
            Backward(_target);
        }

        public void StopTrain()
        {
            if (_trainX != null)
            {
                foreach (var tensor in _trainX)
                {
                    tensor.Release();
                }
            }
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