using Blue.Graph;
using Blue.Operates;

namespace Blue.Kit
{
    public static class Layer
    {

        public static IGraphNode DenseLayer(IGraphNode input, int size, string activation)
        {
            var weight = new DataNode(size * input.GetOutput().count, true);
            XavierOperate.WeightInit(weight.GetOutput(), activation, input.GetOutput().count, size);
            var dot = new DotNode(input, weight);
            var bias = new DataNode(size, true);
            var add = new AddNode(dot, bias);
            return activation switch
            {
                "relu" => new ReLUNode(add),
                "sigmoid" => new SigmoidNode(add),
                _ => add
            };
        }
        
    }
}