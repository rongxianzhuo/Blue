using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalNode : Tensor
    {

        public readonly Tensor Gradient;
        public readonly ComputationalGraph Graph;

        private readonly Operate _clearGradientOp;
        private readonly ComputationalNode[] _inputNodes;
        private readonly List<Operate> _forwardOperates = new List<Operate>();
        private readonly List<Operate> _backwardOperates = new List<Operate>();
        private readonly HashSet<Tensor> _tempTensors = new HashSet<Tensor>();

        public IReadOnlyList<ComputationalNode> InputNodes => _inputNodes;

        internal ComputationalNode(ComputationalGraph graph, ComputationalNode[] inputNodes, params int[] shape) : base(shape)
        {
            _inputNodes = inputNodes ?? Array.Empty<ComputationalNode>();
            Graph = graph;
            Gradient = CreateTempTensor(shape);
            _clearGradientOp = Op.Clear(Gradient, 0f);
        }

        internal Tensor CreateTempTensor(params int[] shape)
        {
            var tensor = new Tensor(shape);
            _tempTensors.Add(tensor);
            return tensor;
        }

        public void AddForwardOperate(Operate operate)
        {
            _forwardOperates.Add(operate);
        }

        public void AddBackwardOperate(Operate operate)
        {
            _backwardOperates.Add(operate);
        }

        public void Backward()
        {
            foreach (var o in _backwardOperates)
            {
                o.Dispatch();
            }

            foreach (var node in _inputNodes)
            {
                node.Backward();
            }
        }

        public void Forward()
        {
            foreach (var o in _forwardOperates)
            {
                o.Dispatch();
            }
        }

        public void ClearGradient()
        {
            _clearGradientOp.Dispatch();
        }

        public override void Dispose()
        {
            _clearGradientOp.Dispose();
            foreach (var o in _forwardOperates)
            {
                o.Dispose();
            }
            _forwardOperates.Clear();
            foreach (var o in _backwardOperates)
            {
                o.Dispose();
            }
            _backwardOperates.Clear();

            foreach (var t in _tempTensors)
            {
                t.Dispose();
            }
            _tempTensors.Clear();
            base.Dispose();
        }
    }
}