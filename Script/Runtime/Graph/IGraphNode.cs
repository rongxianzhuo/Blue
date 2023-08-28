using System;
using UnityEngine;

namespace Blue.Graph
{
    public interface IGraphNode
    {

        ComputeBuffer GetOutput();

        ComputeBuffer GetGradient();

        void Forward();

        void Backward();

        void Destroy();

        void ForeachInputNode(Action<IGraphNode> action);

    }
}