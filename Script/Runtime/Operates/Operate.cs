using UnityEngine;

namespace Blue.Operates
{
    public sealed class Operate
    {

        private readonly ComputeShader _cs;
        private readonly int _kernel;
        private readonly Vector3Int _threadGroupSize;
        private readonly int[] _propertyIds;

        public Operate(string name, string kernel, params string[] propertyNames)
        {
            _cs = Resources.Load<ComputeShader>($"Blue/Shader/{name}");
            _kernel = _cs.FindKernel(kernel);
            _cs.GetKernelThreadGroupSizes(_kernel, out var x, out var y, out var z);
            _threadGroupSize = new Vector3Int((int)x, (int)y, (int)z);
            _propertyIds = new int[propertyNames.Length + 3];
            for (var i = 0; i < propertyNames.Length; i++)
            {
                _propertyIds[i] = Shader.PropertyToID(propertyNames[i]);
            }
            _propertyIds[_propertyIds.Length - 3] = Shader.PropertyToID("total_thread_x");
            _propertyIds[_propertyIds.Length - 2] = Shader.PropertyToID("total_thread_y");
            _propertyIds[_propertyIds.Length - 1] = Shader.PropertyToID("total_thread_z");
        }

        public OperateHandler CreateTask()
        {
            return new OperateHandler(_kernel, _threadGroupSize, _cs, _propertyIds);
        }

    }
}