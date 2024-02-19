using Blue.Core;
using Blue.Graph;

namespace Blue.Runtime.NN
{
    public class Dropout : Module
    {

        private readonly float _dropout;

        public Dropout(float dropout)
        {
            _dropout = dropout;
        }
        
        public override ComputationalNode CreateGraph(params ComputationalNode[] input)
        {
            var node = input[0];
            var dropoutNode = new ComputationalNode(new []{node}, node.Size);
            var weightArray = new float[node.FlattenSize];
            var weight = dropoutNode.CreateTempTensor(node.Size);
            dropoutNode.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= _dropout ? 1f : 0f;
                }
                weight.SetData(weightArray);
            }));
            dropoutNode.AddForwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", node)
                .SetTensor("b", weight)
                .SetTensor("result", dropoutNode)
                .SetDispatchSize(node.FlattenSize));
            dropoutNode.AddBackwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", dropoutNode.Gradient)
                .SetTensor("b", weight)
                .SetTensor("result", node.Gradient)
                .SetDispatchSize(node.Gradient.FlattenSize));

            return dropoutNode;
        }
    }
}