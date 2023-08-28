using System;
using UnityEngine;


using Blue.Core;namespace Blue.Graph
{
    public class MatMulNode : IGraphNode
    {
        
        private static Operate _matMulOp;

        private static Operate GetMatMulOp() => _matMulOp ??= new Operate("Common/MatMul", "CSMain"
            , "wl"
            , "wr"
            , "left"
            , "right"
            , "result");

        private readonly IGraphNode _left;
        private readonly IGraphNode _right;
        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;
        
        public MatMulNode(IGraphNode left, IGraphNode right)
        {
            _left = left;
            _right = right;
            var size = right.GetOutput().count / left.GetOutput().count;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }

        public ComputeBuffer GetOutput() => _output;

        public ComputeBuffer GetGradient() => _gradient;

        public void Forward()
        {
            GetMatMulOp().CreateTask()
                .SetInt(_left.GetOutput().count)
                .SetInt(_output.count)
                .SetBuffer(_left.GetOutput())
                .SetBuffer(_right.GetOutput())
                .SetBuffer(_output)
                .Dispatch(new Vector3Int(_output.count, 1, 1));
        }

        public void Backward()
        {
            GetMatMulOp().CreateTask()
                .SetInt(_output.count)
                .SetInt(1)
                .SetBuffer(_right.GetOutput())
                .SetBuffer(_gradient)
                .SetBuffer(_left.GetGradient())
                .Dispatch(new Vector3Int(_left.GetGradient().count, 1, 1));
            GetMatMulOp().CreateTask()
                .SetInt(1)
                .SetInt(_output.count)
                .SetBuffer(_left.GetOutput())
                .SetBuffer(_gradient)
                .SetBuffer(_right.GetGradient())
                .Dispatch(new Vector3Int(_right.GetGradient().count, 1, 1));
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
            action(_left);
            action(_right);
        }
    }
}