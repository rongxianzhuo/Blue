using System;
using Blue.Core;
using UnityEngine;

namespace Blue.Graph
{
    public interface IGraphNode
    {

        Tensor GetOutput();

        Tensor GetGradient();

        void Forward();

        void Backward();

        void Destroy();

        void ForeachInputNode(Action<IGraphNode> action);

    }
}