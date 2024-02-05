using System;
using UnityEngine;
using Object = UnityEngine.Object;

namespace Blue.Core
{
    public class Operate : IDisposable
    {
        
        private readonly ComputeShader _cs;
        private readonly int _kernel;
        private readonly System.Action _dispatchAction;
        private int _groupSizeX;
        private int _groupSizeY;
        private int _groupSizeZ;

        public static int PropertyId(string propertyName) => Shader.PropertyToID(propertyName);

        public Operate(System.Action dispatchAction)
        {
            _dispatchAction = dispatchAction;
        }

        public Operate(string name, string kernel)
        {
            _cs = Object.Instantiate(Resources.Load<ComputeShader>($"Blue/Shader/{name}"));
            _kernel = _cs.FindKernel(kernel);
        }

        public Operate SetInt(int id, int i)
        {
            _cs.SetInt(id, i);
            return this;
        }

        public Operate SetInt(string name, int i)
        {
            _cs.SetInt(name, i);
            return this;
        }

        public Operate SetFloat(int id, float f)
        {
            _cs.SetFloat(id, f);
            return this;
        }

        public Operate SetFloat(string name, float f)
        {
            _cs.SetFloat(name, f);
            return this;
        }

        public Operate SetTensor(string name, Tensor tensor)
        {
            tensor.SetToShader(_cs, _kernel, name);
            return this;
        }

        public Operate SetDispatchSize(int x, int y=1, int z=1)
        {
            _cs.GetKernelThreadGroupSizes(_kernel, out var gx, out var gy, out var gz);
            _groupSizeX = x / (int) gx;
            if (x % gx != 0) _groupSizeX++;
            _groupSizeY = y / (int) gy;
            if (y % gy != 0) _groupSizeY++;
            _groupSizeZ = z / (int) gz;
            if (z % gz != 0) _groupSizeZ++;
            return this;
        }

        public Operate Dispatch()
        {
            if (_dispatchAction == null)
            {
                _cs.Dispatch(_kernel, _groupSizeX, _groupSizeY, _groupSizeZ);
            }
            else
            {
                _dispatchAction();
            }
            return this;
        }

        public void Dispose()
        {
#if UNITY_EDITOR
            Object.DestroyImmediate(_cs);
#else
            Object.Destroy(_cs);
#endif
        }
    }
}