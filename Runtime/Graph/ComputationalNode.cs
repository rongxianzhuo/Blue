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

        public Tensor Gradient => IsView ? _viewOrigin.Gradient : _gradient;

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
            if (IsParameter) GetOrCreateGradient();
        }

        public ComputationalNode(IReadOnlyList<int> shape, ComputationalNode origin) : base(shape, origin)
        {
            _viewOrigin = origin;
            IsParameter = false;
            _inputNodes.Add(origin);
        }

        public ComputationalNode View(params int[] shape) => new ComputationalNode(shape, this);

        public ComputationalGraph Graph(params ComputationalNode[] inputs) => new ComputationalGraph(this, inputs);

        public Tensor GetOrCreateGradient()
        {
            if (IsView) return _viewOrigin.GetOrCreateGradient();
            if (_gradient != null) return Gradient;
            _gradient = CreateTempTensor(Size);
            _clearGradientOp = Op.Clear(_gradient, 0f);
            return _gradient;
        }

        internal Tensor CreateTempTensor(params int[] shape)
        {
            var tensor = new Tensor(shape);
            _tempTensors.Add(tensor);
            return tensor;
        }

        public void AddInputNode(ComputationalNode node)
        {
            if (node.Gradient != null) GetOrCreateGradient();
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