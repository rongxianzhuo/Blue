using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Optimizer/Adam", "CSMain"
            , "t", "beta1", "beta2", "learning_rate", "g", "m", "v", "theta");

        public float LearningRate = 0.001f;
        
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly List<Tensor> _m = new List<Tensor>();
        private readonly List<Tensor> _v = new List<Tensor>();
        private readonly List<float> _t = new List<float>();
        
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

            while (_t.Count <= node.Id) _t.Add(0);
            var t = _t[node.Id];
            t++;
            _t[node.Id] = t;
            GetOperate().CreateTask()
                .SetFloat(t)
                .SetFloat(_beta1)
                .SetFloat(_beta2)
                .SetFloat(LearningRate)
                .SetTensor(gradient)
                .SetTensor(m)
                .SetTensor(v)
                .SetTensor(param)
                .Dispatch(param.FlattenSize);
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
        }
    }
}