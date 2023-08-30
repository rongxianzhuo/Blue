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
        
        public SimpleModel(TensorNode input, IGraphNode outputNode) : base(outputNode)
        {
            _input = input;
            _target = new ComputeBuffer(outputNode.GetOutput().count, 4);
            _outputArray = new float[outputNode.GetOutput().count];
        }

        public void Train(float[] x, float[] y)
        {
            _input.GetOutput().SetData(x);
            _target.SetData(y);
            Forward();
            Backward(_target);
        }

        public override void Destroy()
        {
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