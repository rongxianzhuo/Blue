using System.Linq;
using Blue.Graph;
using Blue.Operates;
using Blue.Util;
using UnityEngine;

namespace Blue.Kit
{
    public static class LossFunction
    {

        public static void L2Loss(IGraphNode node, ComputeBuffer target)
        {
            CopyOperate.Calculate(node.GetOutput(), 0, node.GetGradient(), 0);
            AddOperate.Calculate(node.GetGradient(), target, -1, 0);
        }

        public static void CrossEntropyLoss(IGraphNode node, ComputeBuffer target)
        {
            CrossEntropyLossOperate.Calculate(node.GetOutput(), target, node.GetGradient());
        }
        
    }
}