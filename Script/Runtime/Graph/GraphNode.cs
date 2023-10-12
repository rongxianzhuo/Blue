using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public abstract class GraphNode
    {

        public IReadOnlyList<GraphNode> ReadOnlyInputNodes => InputNodes;

        protected readonly List<GraphNode> InputNodes = new List<GraphNode>();
        protected readonly List<Operate> ForwardOperates = new List<Operate>();
        protected readonly List<Operate> BackwardOperates = new List<Operate>();

        private readonly HashSet<Tensor> _bindTensors = new HashSet<Tensor>();

        public void AddInputNode(params ComputationalNode[] node)
        {
            InputNodes.AddRange(node);
        }

        public void AddForwardOperate(Operate operate)
        {
            ForwardOperates.Add(operate);
        }

        public void AddBackwardOperate(Operate operate)
        {
            BackwardOperates.Add(operate);
        }

        public Tensor CreateTensor(params int[] shape)
        {
            var tensor = new Tensor(shape);
            _bindTensors.Add(tensor);
            return tensor;
        }

        public void Forward()
        {
            foreach (var o in ForwardOperates)
            {
                o.Dispatch();
            }
        }

        public void Backward()
        {
            foreach (var o in BackwardOperates)
            {
                o.Dispatch();
            }

            foreach (var node in InputNodes)
            {
                node.Backward();
            }
        }

        public void Destroy()
        {
            foreach (var o in ForwardOperates)
            {
                o.Destroy();
            }
            ForwardOperates.Clear();
            foreach (var o in BackwardOperates)
            {
                o.Destroy();
            }
            BackwardOperates.Clear();

            foreach (var t in _bindTensors)
            {
                t.Release();
            }
            _bindTensors.Clear();
        }

    }
}