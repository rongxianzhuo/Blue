using UnityEngine;

namespace Blue.Core
{
    public sealed class Operate
    {
        public struct OperateHandler
        {
            
            private readonly int _kernel;
            private readonly ComputeShader _cs;
            private readonly Vector3Int _threadNumber;
            private readonly int[] _propertyIds;
            
            private int _propertyIndex;

            public OperateHandler(int kernel, Vector3Int threadNumber, ComputeShader cs, int[] propertyIds)
            {
                _cs = cs;
                _threadNumber = threadNumber;
                _kernel = kernel;
                _propertyIds = propertyIds;
                _propertyIndex = 0;
            }

            public OperateHandler SetInt(int i)
            {
                _cs.SetInt(_propertyIds[_propertyIndex++], i);
                return this;
            }

            public OperateHandler SetFloat(float f)
            {
                _cs.SetFloat(_propertyIds[_propertyIndex++], f);
                return this;
            }

            public OperateHandler SetBuffer(ComputeBuffer buffer)
            {
                _cs.SetBuffer(_kernel, _propertyIds[_propertyIndex++], buffer);
                return this;
            }

            public void Dispatch(Vector3Int size)
            {
                _cs.SetInt(_propertyIds[_propertyIndex++], size.x);
                _cs.SetInt(_propertyIds[_propertyIndex++], size.y);
                _cs.SetInt(_propertyIds[_propertyIndex++], size.z);
                var groupX = size.x / _threadNumber.x;
                if (size.x % _threadNumber.x != 0) groupX++;
                var groupY = size.y / _threadNumber.y;
                if (size.y % _threadNumber.y != 0) groupY++;
                var groupZ = size.z / _threadNumber.z;
                if (size.z % _threadNumber.z != 0) groupZ++;
                _cs.Dispatch(_kernel, groupX, groupY, groupZ);
            }
        }

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