using System.Linq;
using Blue.Graph;
using Blue.Operates;
using Blue.Util;
using UnityEngine;

namespace Blue.Kit
{
    public static class LossFunction
    {

        private static Operate _l2Loss;

        private static Operate GetL2LossOp() => _l2Loss ??= new Operate("L2Loss", "CSMain"
            , "output", "target", "gradient");

        private static Operate _crossEntropyLoss;

        private static Operate GetCrossEntropyLossOp() => _crossEntropyLoss ??= new Operate("CrossEntropyLoss", "CSMain"
            , "r_buffer1", "r_buffer2", "rw_buffer1");

        public static void L2Loss(IGraphNode node, ComputeBuffer target)
        {
            GetL2LossOp().CreateTask()
                .SetBuffer(node.GetOutput())
                .SetBuffer(target)
                .SetBuffer(node.GetGradient())
                .Dispatch(new Vector3Int(target.count, 1, 1));
        }

        public static void CrossEntropyLoss(IGraphNode node, ComputeBuffer target)
        {
            GetCrossEntropyLossOp().CreateTask()
                .SetBuffer(node.GetOutput())
                .SetBuffer(target)
                .SetBuffer(node.GetGradient())
                .Dispatch(new Vector3Int(target.count, 1, 1));
        }
        
    }
}