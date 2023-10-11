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

        public abstract Tensor GetOutput();

        public abstract Tensor GetGradient();

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
            OnDestroy();
            foreach (var o in ForwardOperates)
            {
                o.Destroy();
            }
            foreach (var o in BackwardOperates)
            {
                o.Destroy();
            }
        }

        protected virtual void OnDestroy()
        {
            
        }

    }
}