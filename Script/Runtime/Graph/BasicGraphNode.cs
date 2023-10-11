using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public abstract class BasicGraphNode : GraphNode
    {
        
        private readonly List<Operate> _forwardOperate = new List<Operate>();
        private readonly List<Operate> _backwardOperate = new List<Operate>();
        
        private Tensor _output;
        private Tensor _gradient;
        private List<GraphNode> _inputNodes;

        protected abstract void GetOutputSize(out int batchSize, out int size);
        
        protected abstract void OnDestroy();

        protected abstract void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward);

        public override Tensor GetOutput()
        {
            if (_output == null)
            {
                GetOutputSize(out var batchSize, out var size);
                _output = new Tensor(batchSize, size);
            }

            return _output;
        }

        public override Tensor GetGradient()
        {
            if (_gradient == null)
            {
                GetOutputSize(out var batchSize, out var size);
                _gradient = new Tensor(batchSize, size);
            }

            return _gradient;
        }

        public override void Forward()
        {
            GetOutputSize(out var batchSize, out var size);
            if (_forwardOperate.Count == 0)
            {
                UpdateOperate(batchSize, _forwardOperate, _backwardOperate);
            }
            foreach (var op in _forwardOperate)
            {
                op.Dispatch();
            }
        }

        public override void Backward()
        {
            foreach (var op in _backwardOperate)
            {
                op.Dispatch();
            }

            if (_inputNodes == null)
            {
                _inputNodes = new List<GraphNode>();
                ForeachInputNode(node => _inputNodes.Add(node));
            }

            foreach (var node in _inputNodes)
            {
                node.Backward();
            }
        }

        public override void Destroy()
        {
            OnDestroy();
            _output?.Release();
            _gradient?.Release();
            foreach (var op in _forwardOperate)
            {
                op.Destroy();
            }
            foreach (var op in _backwardOperate)
            {
                op.Destroy();
            }
            _forwardOperate.Clear();
            _backwardOperate.Clear();
        }
    }
}