using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public class SimpleModel : Model
    {

        private readonly TensorNode _input;
        private readonly ComputeBuffer _target;
        private readonly float[] _outputArray;

        private ComputeBuffer _trainX;
        private ComputeBuffer _trainY;
        
        public SimpleModel(TensorNode input, IGraphNode outputNode) : base(outputNode)
        {
            _input = input;
            _target = new ComputeBuffer(outputNode.GetOutput().count, 4);
            _outputArray = new float[outputNode.GetOutput().count];
        }

        public void StartTrain(List<float> x, List<float> y)
        {
            StopTrain();
            _trainX = new ComputeBuffer(x.Count, 4);
            _trainX.SetData(x);
            _trainY = new ComputeBuffer(y.Count, 4);
            _trainY.SetData(y);
        }

        public void UpdateTrain(int sampleIndex)
        {
            var inputLength = _input.GetOutput().count;
            var outputLength = _target.count;
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