using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Graph
{
    public class ComputationalNode : Tensor
    {

        public readonly bool IsParameter;

        private readonly List<Operate> _forwardOperates = new List<Operate>();
        private readonly List<Operate> _backwardOperates = new List<Operate>();
        private readonly HashSet<Tensor> _tempTensors = new HashSet<Tensor>();
        private readonly List<ComputationalNode> _inputNodes = new List<ComputationalNode>();
        
        private Operate _clearGradientOp;

        public IReadOnlyList<ComputationalNode> InputNodes => _inputNodes;

        public Tensor Gradient { get; private set; }

        public ComputationalNode(ComputationalNode[] inputNodes, params int[] shape) : base(shape)
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

        public Tensor GetOrCreateGradient()
        {
            if (Gradient != null) return Gradient;
            Gradient = CreateTempTensor(Size);
            _clearGradientOp = Op.Clear(Gradient, 0f);
            return Gradient;
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

        public void Forward()
        {
            foreach (var o in _forwardOperates)
            {
                o.Dispatch();
            }
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

        public static ComputationalNode operator *(ComputationalNode a, ComputationalNode b)
        {
            var size = a.FlattenSize > b.FlattenSize ? a.Size : b.Size;
            var c = new ComputationalNode(new[] { a, b }, size);
            c.AddForwardOperate(new Operate("NN/Mul/Forward", "CSMain")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetTensor("a", a)
                .SetTensor("b", b)
                .SetTensor("c", c)
                .SetDispatchSize(c.FlattenSize));
            if (a.Gradient != null) c.AddBackwardOperate(new Operate("NN/Mul/BackwardA", "CSMain")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetInt("c_len", c.FlattenSize)
                .SetTensor("a_gradient", a.Gradient)
                .SetTensor("b", b)
                .SetTensor("c_gradient", c.Gradient)
                .SetDispatchSize(a.FlattenSize));
            if (b.Gradient != null) c.AddBackwardOperate(new Operate("NN/Mul/BackwardB", "CSMain")
                .SetInt("a_len", a.FlattenSize)
                .SetInt("b_len", b.FlattenSize)
                .SetInt("c_len", c.FlattenSize)
                .SetTensor("a", a)
                .SetTensor("b_gradient", b.Gradient)
                .SetTensor("c_gradient", c.Gradient)
                .SetDispatchSize(b.FlattenSize));
            return c;
        }
    }
}