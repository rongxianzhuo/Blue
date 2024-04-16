using Blue.Graph;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    public static class Common
    {
        
        public static float RandN(float a, float v)
        {
            var u1 = Random.Range(0f, 1f);
            var u2 = Random.Range(0f, 1f);
            var n = Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Cos(2 * Mathf.PI * u2);
            return n * v + a;
        }
        
        public static ComputationalNode Concat(params ComputationalNode[] nodes)
        {
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.Size[1];
            }
            var concat = new ComputationalNode(nodes, nodes[0].Size[0], size);
            
            var start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t;
                concat.AddForwardOperate(Op.Copy(inputNode, 0, 0
                    , concat, start, size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t.Gradient;
                if (inputNode != null) concat.AddBackwardOperate(Op.Copy(concat.Gradient, start, size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += t.Size[1];
            }

            return concat;
        }
    }
}