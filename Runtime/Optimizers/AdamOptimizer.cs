using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        private readonly int _tId = Operate.PropertyId("t");
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly HashSet<Operate> _op = new HashSet<Operate>();
        private readonly HashSet<Tensor> _tensors = new HashSet<Tensor>();

        private float _t;

        public AdamOptimizer(IEnumerable<ComputationalNode> nodes, float learningRate=0.001f, float weightDecay=0f)
        {
            foreach (var node in nodes)
            {
                AddParameter(node, learningRate, weightDecay);
            }
        }

        public AdamOptimizer(IEnumerable<Runtime.NN.Module> modules, float learningRate=0.001f, float weightDecay=0f)
        {
            foreach (var module in modules)
            {
                foreach (var node in module.GetAllParameters())
                {
                    AddParameter(node, learningRate, weightDecay);
                }
            }
        }

        private void AddParameter(ComputationalNode node, float learningRate, float weightDecay)
        {
            var gradient = node.Gradient;
            var m = new Tensor(node.Size);
            var v = new Tensor(node.Size);
            _tensors.Add(m);
            _tensors.Add(v);
            var op = new Operate("Optimizer/Adam", "CSMain")
                .SetFloat("t", 0f)
                .SetFloat("beta1", _beta1)
                .SetFloat("beta2", _beta2)
                .SetFloat("weight_decay", weightDecay)
                .SetFloat("learning_rate", learningRate)
                .SetTensor("g", gradient)
                .SetTensor("m", m)
                .SetTensor("v", v)
                .SetTensor("theta", node)
                .SetDispatchSize(node.FlattenSize);
            _op.Add(op);
        }

        public void Step()
        {
            _t++;
            foreach (var op in _op)
            {
                op.SetFloat(_tId, _t);
                op.Dispatch();
            }
        }

        public void Dispose()
        {
            foreach (var t in _tensors)
            {
                t?.Dispose();
            }
            _tensors.Clear();

            foreach (var op in _op)
            {
                op?.Dispose();
            }
            _op.Clear();
        }
    }
}