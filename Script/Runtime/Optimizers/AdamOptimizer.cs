using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        public float LearningRate = 0.001f;

        private readonly int _tId = OperateInstance.PropertyId("t");
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly List<Tensor> _m = new List<Tensor>();
        private readonly List<Tensor> _v = new List<Tensor>();
        private readonly List<float> _t = new List<float>();
        private readonly List<OperateInstance> _op = new List<OperateInstance>();
        
        public void Step(TensorNode node)
        {
            var param = node.GetOutput();
            var gradient = node.GetGradient();
            while (_m.Count <= node.Id) _m.Add(null);
            var m = _m[node.Id];
            if (m == null)
            {
                m = new Tensor(param.Size);
                _m[node.Id] = m;
            }
            
            while (_v.Count <= node.Id) _v.Add(null);
            var v = _v[node.Id];
            if (v == null)
            {
                v = new Tensor(param.Size);
                _v[node.Id] = v;
            }
            
            while (_op.Count <= node.Id) _op.Add(null);
            var op = _op[node.Id];
            if (op == null)
            {
                op = new OperateInstance("Optimizer/Adam", "CSMain")
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

            while (_t.Count <= node.Id) _t.Add(0);
            var t = _t[node.Id];
            t++;
            _t[node.Id] = t;
            op.SetFloat(_tId, t);
            op.Dispatch();
        }

        public void Destroy()
        {
            foreach (var m in _m)
            {
                m?.Release();
            }
            foreach (var v in _v)
            {
                v?.Release();
            }

            foreach (var op in _op)
            {
                op?.Destroy();
            }
        }
    }
}