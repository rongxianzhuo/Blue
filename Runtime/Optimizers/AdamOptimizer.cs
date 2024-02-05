using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        public float LearningRate = 0.001f;

        private readonly int _tId = Operate.PropertyId("t");
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly List<Operate> _op = new List<Operate>();
        private readonly HashSet<Tensor> _tensors = new HashSet<Tensor>();

        private float _t;

        public void Step(IEnumerable<ComputationalNode> nodes)
        {
            _t++;
            foreach (var node in nodes)
            {
                while (_op.Count <= node.Id) _op.Add(null);
                var op = _op[node.Id];
                if (op == null)
                {
                    var param = node;
                    var gradient = node.TotalGradient;
                    var m = new Tensor(param.Size);
                    var v = new Tensor(param.Size);
                    _tensors.Add(m);
                    _tensors.Add(v);
                    op = new Operate("Optimizer/Adam", "CSMain")
                        .SetFloat("t", 0f)
                        .SetFloat("beta1", _beta1)
                        .SetFloat("beta2", _beta2)
                        .SetFloat("learning_rate", LearningRate)
                        .SetTensor("g", gradient)
                        .SetTensor("m", m)
                        .SetTensor("v", v)
                        .SetTensor("theta", param)
                        .SetDispatchSize(param.FlattenSize);
                    _op[node.Id] = op;
                }

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