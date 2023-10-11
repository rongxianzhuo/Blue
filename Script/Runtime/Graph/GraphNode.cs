using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public abstract class GraphNode
    {

        public IReadOnlyList<GraphNode> ReadOnlyInputNodes => InputNodes;

        protected readonly List<GraphNode> InputNodes = new List<GraphNode>();

        public abstract Tensor GetOutput();

        public abstract Tensor GetGradient();

        public abstract void Forward();

        public abstract void Backward();

        public abstract void Destroy();

    }
}