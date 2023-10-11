using System;
using Blue.Core;

namespace Blue.Graph
{
    public abstract class GraphNode
    {

        public abstract Tensor GetOutput();

        public abstract Tensor GetGradient();

        public abstract void Forward();

        public abstract void Backward();

        public abstract void Destroy();

        public abstract void ForeachInputNode(Action<GraphNode> action);

    }
}