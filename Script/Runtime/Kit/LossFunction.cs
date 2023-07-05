using System.Linq;
using Blue.Graph;
using Blue.Operates;
using Blue.Util;
using UnityEngine;

namespace Blue.Kit
{
    public static class LossFunction
    {

        public static float L2Loss(IGraphNode node, ComputeBuffer target, bool getLoss)
        {
            CopyOperate.Calculate(node.GetOutput(), node.GetGradient());
            AddOperate.Calculate(node.GetGradient(), target, -1, 0);
            if (!getLoss) return 0;
            var array = FloatArrayPool.Default.Get(node.GetGradient());
            var sum = 0f;
            foreach (var f in array)
            {
                sum += f * f;
            }
            sum *= 0.5f;
            sum /= target.count;
            FloatArrayPool.Default.Recycle(array);
            return sum;
        }
        
    }
}