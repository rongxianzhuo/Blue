using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Graph
{
    public abstract class BasicGraphNode : IGraphNode
    {
        
        private readonly List<Operate> _forwardOperate = new List<Operate>();
        private readonly List<Operate> _backwardOperate = new List<Operate>();
        
        private Tensor _output;
        private Tensor _gradient;

        protected abstract void GetOutputSize(out int batchSize, out int size);
        
        protected abstract void OnDestroy();

        public abstract void ForeachInputNode(Action<IGraphNode> action);

        protected abstract void UpdateOperate(int batchSize, List<Operate> forward, List<Operate> backward);

        public Tensor GetOutput()
        {
            if (_output == null)
            {
                GetOutputSize(out var batchSize, out var size);
                _output = new Tensor(batchSize, size);
            }

            return _output;
        }

        public Tensor GetGradient()
        {
            if (_gradient == null)
            {
                GetOutputSize(out var batchSize, out var size);
                _gradient = new Tensor(batchSize, size);
            }

            return _gradient;
        }

        public void Forward()
        {
            GetOutputSize(out var batchSize, out var size);
            if (_forwardOperate.Count == 0 || batchSize != GetOutput().Size[0])
            {
                GetOutput().Resize(batchSize, size);
                GetGradient().Resize(batchSize, size);
                foreach (var op in _forwardOperate)
                {
                    op.Destroy();
                }
                _forwardOperate.Clear();
                foreach (var op in _backwardOperate)
                {
                    op.Destroy();
                }
                _backwardOperate.Clear();
                UpdateOperate(batchSize, _forwardOperate, _backwardOperate);
            }
            foreach (var op in _forwardOperate)
            {
                op.Dispatch();
            }
        }

        public void Backward()
        {
            foreach (var op in _backwardOperate)
            {
                op.Dispatch();
            }
        }

        public void Destroy()
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