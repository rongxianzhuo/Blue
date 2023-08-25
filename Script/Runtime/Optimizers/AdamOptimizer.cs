using System.Collections.Generic;
using Blue.Operates;
using UnityEngine;

namespace Blue.Optimizers
{
    public class AdamOptimizer : IOptimizer
    {

        private static Operate _operate;

        private static Operate GetOperate() => _operate ??= new Operate("Adam", "CSMain"
            , "t", "beta1", "beta2", "learning_rate", "g", "m", "v", "theta");

        public float LearningRate = 0.001f;
        
        private readonly float _beta1 = 0.9f;
        private readonly float _beta2 = 0.999f;
        private readonly Dictionary<ComputeBuffer, ComputeBuffer> _m = new Dictionary<ComputeBuffer, ComputeBuffer>();
        private readonly Dictionary<ComputeBuffer, ComputeBuffer> _v = new Dictionary<ComputeBuffer, ComputeBuffer>();
        private readonly Dictionary<ComputeBuffer, float> _t = new Dictionary<ComputeBuffer, float>();
        
        public void OnBackwardPropagation(ComputeBuffer param, ComputeBuffer gradient)
        {
            if (!_m.TryGetValue(param, out var m))
            {
                m = new ComputeBuffer(param.count, 4);
                _m[param] = m;
            }
            if (!_v.TryGetValue(param, out var v))
            {
                v = new ComputeBuffer(param.count, 4);
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
                .SetBuffer(gradient)
                .SetBuffer(m)
                .SetBuffer(v)
                .SetBuffer(param)
                .Dispatch(new Vector3Int(param.count, 1, 1));
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