using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public partial class ComputationalNode : Tensor
    {

        public readonly bool IsParameter;

        private readonly ComputationalNode _viewOrigin;
        private readonly List<Operate> _forwardOperates = new List<Operate>();
        private readonly List<Operate> _backwardOperates = new List<Operate>();
        private readonly HashSet<Tensor> _tempTensors = new HashSet<Tensor>();
        private readonly List<ComputationalNode> _inputNodes = new List<ComputationalNode>();

        private Tensor _gradient;
        private Operate _clearGradientOp;

        public IReadOnlyList<ComputationalNode> InputNodes => _inputNodes;

        public Tensor Gradient => _gradient ??= CreateGradient();

        public ComputationalNode(IEnumerable<ComputationalNode> inputNodes, params int[] shape) : base(shape)
        {
            foreach (var node in inputNodes)
            {
                AddInputNode(node);
            }
        }

        public ComputationalNode(bool isParameter, params int[] shape) : base(shape)
        {
            IsParameter = isParameter;
        }

        public ComputationalNode(IReadOnlyList<int> shape
            , ComputationalNode origin
            , IReadOnlyList<int> stride=null) : base(shape, origin, stride)
        {
            _viewOrigin = origin;
            IsParameter = false;
            _inputNodes.Add(origin);
        }

        public ComputationalNode View(params int[] shape) => new ComputationalNode(shape, this);

        public ComputationalGraph Graph(params ComputationalNode[] inputs) => new ComputationalGraph(this, inputs);

        private Tensor CreateGradient()
        {
            var originGradient = _viewOrigin?.Gradient;
            var gradient = new Tensor(Size, originGradient, StrideInMemory);
            _tempTensors.Add(gradient);
            _clearGradientOp = Op.Clear(gradient, 0f);
            return gradient;
        }

        public void AddInputNode(ComputationalNode node)
        {
            _inputNodes.Add(node);
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

            if (_inputNodes.Count <= 0) return;
            foreach (var node in _inputNodes)
            {
                if (node.Gradient != null) node.Backward();
            }
        }

        public ComputationalNode Forward()
        {
            foreach (var o in _forwardOperates)
            {
                o.Dispatch();
            }
            return this;
        }

        public ComputationalNode Forward(NN.Module module)
        {
            return module.Build(this);
        }

        public void ClearGradient()
        {
            _clearGradientOp?.Dispatch();
        }

        public override void Dispose()
        {
            _clearGradientOp?.Dispose();
            _clearGradientOp = null;
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