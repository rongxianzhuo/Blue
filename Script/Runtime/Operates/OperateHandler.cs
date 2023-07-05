using UnityEngine;

namespace Blue.Operates
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
}