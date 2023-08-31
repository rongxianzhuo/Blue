using System.Collections.Generic;
using UnityEngine;


using Blue.Core;namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Optimizer/Adam", "CSMain"
            , "t", "beta1", "beta2", "learning_rate", "g", "m", "v", "theta");

        public float LearningRate = 0.001f;
        
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly Dictionary<Tensor, Tensor> _m = new Dictionary<Tensor, Tensor>();
        private readonly Dictionary<Tensor, Tensor> _v = new Dictionary<Tensor, Tensor>();
        private readonly Dictionary<Tensor, float> _t = new Dictionary<Tensor, float>();
        
        public void Step(Tensor param, Tensor gradient)
        {
            if (!_m.TryGetValue(param, out var m))
            {
                m = new Tensor(param.Size);
                _m[param] = m;
            }
            if (!_v.TryGetValue(param, out var v))
            {
                v = new Tensor(param.Size);
                _v[param] = v;
            }

            if (!_t.TryGetValue(param, out var t))
            {
                t = 0f;
            }
            t++;
            _t[param] = t;
            GetOperate().CreateTask()
                .SetFloat(t)
                .SetFloat(_beta1)
                .SetFloat(_beta2)
                .SetFloat(LearningRate)
                .SetTensor(gradient)
                .SetTensor(m)
                .SetTensor(v)
                .SetTensor(param)
                .Dispatch(new Vector3Int(param.Size, 1, 1));
        }

        public void Destroy()
        {
            foreach (var m in _m.Values)
            {
                m.Release();
            }
            foreach (var v in _v.Values)
            {
                v.Release();
            }
        }
    }
}