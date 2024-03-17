using Blue.Core;
using Blue.Graph;

namespace Blue.NN
{
    public class Dropout : Module
    {

        private readonly float _dropout;

        public Dropout(float dropout)
        {
            _dropout = dropout;
        }
        
        public override ComputationalNode Build(params ComputationalNode[] input)
        {
            var node = input[0];
            var weightArray = new float[node.FlattenSize];
            var dropout = new ComputationalNode(false, node.Size);
            dropout.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= _dropout ? 1f : 0f;
                }
                dropout.SetData(weightArray);
            }));
            return node * dropout;
        }
    }
}