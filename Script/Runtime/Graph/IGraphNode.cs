using System;
using UnityEngine;

namespace Blue.Graph
{
    public interface IGraphNode
    {

        ComputeBuffer GetOutput();

        ComputeBuffer GetGradient();

        void Calculate();

        void GradientPropagation();

        void Destroy();

        void ForeachInputNode(Action<IGraphNode> action);

    }
}