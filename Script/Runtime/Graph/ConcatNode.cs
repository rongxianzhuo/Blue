using System;
using Blue.Core;
using UnityEngine;

namespace Blue.Graph
{
    public class ConcatNode : IGraphNode
    {
        
        private static Operate _copyOperate;

        private static Operate GetCopyOperate() => _copyOperate ??= new Operate("Common/Copy", "CSMain"
            , "r_buffer1", "src_offset", "rw_buffer1", "dst_offset");

        private readonly IGraphNode[] _nodes;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public ConcatNode(params IGraphNode[] nodes)
        {
            _nodes = nodes;
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.GetOutput().count;
            }
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }
        
        public ComputeBuffer GetOutput()
        {
            return _output;
        }

        public ComputeBuffer GetGradient()
        {
            return _gradient;
        }

        public void Calculate()
        {
            var i = 0;
            foreach (var node in _nodes)
            {
                GetCopyOperate().CreateTask()
                    .SetBuffer(node.GetOutput())
                    .SetInt(0)
                    .SetBuffer(_output)
                    .SetInt(i)
                    .Dispatch(new Vector3Int(_output.count, 1, 1));
                i += node.GetOutput().count;
            }
        }

        public void GradientPropagation()
        {
            var i = 0;
            foreach (var node in _nodes)
            {
                GetCopyOperate().CreateTask()
                    .SetBuffer(_output)
                    .SetInt(i)
                    .SetBuffer(node.GetGradient())
                    .SetInt(0)
                    .Dispatch(new Vector3Int(node.GetGradient().count, 1, 1));
                i += node.GetOutput().count;
            }
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            foreach (var node in _nodes)
            {
                action(node);
            }
        }
    }
}