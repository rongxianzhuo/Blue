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

            public OperateHandler SetTensor(Tensor tensor)
            {
                tensor.SetToShader(_cs, _kernel, _propertyIds[_propertyIndex++]);
                return this;
            }

            public void Dispatch(int x, int y=1, int z=1)
            {
                var groupX = x / _threadNumber.x;
                if (x % _threadNumber.x != 0) groupX++;
                var groupY = y / _threadNumber.y;
                if (y % _threadNumber.y != 0) groupY++;
                var groupZ = z / _threadNumber.z;
                if (z % _threadNumber.z != 0) groupZ++;
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
            _propertyIds = new int[propertyNames.Length];
            for (var i = 0; i < propertyNames.Length; i++)
            {
                _propertyIds[i] = Shader.PropertyToID(propertyNames[i]);
            }
        }

        public OperateHandler CreateTask()
        {
            return new OperateHandler(_kernel, _threadGroupSize, _cs, _propertyIds);
        }

    }
}