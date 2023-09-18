using UnityEngine;

namespace Blue.Core
{
    public class OperateInstance
    {
        
        private readonly ComputeShader _cs;
        private readonly int _kernel;
        private int _groupSizeX;
        private int _groupSizeY;
        private int _groupSizeZ;

        public OperateInstance(string name, string kernel)
        {
            _cs = Object.Instantiate(Resources.Load<ComputeShader>($"Blue/Shader/{name}"));
            _kernel = _cs.FindKernel(kernel);
        }

        public OperateInstance SetInt(string name, int i)
        {
            _cs.SetInt(name, i);
            return this;
        }

        public OperateInstance SetFloat(string name, float f)
        {
            _cs.SetFloat(name, f);
            return this;
        }

        public OperateInstance SetTensor(string name, Tensor tensor)
        {
            tensor.SetToShader(_cs, _kernel, name);
            return this;
        }

        public OperateInstance SetDispatchSize(int x, int y=1, int z=1)
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

        public void Dispatch()
        {
            _cs.Dispatch(_kernel, _groupSizeX, _groupSizeY, _groupSizeZ);
        }

        public void Destroy()
        {
            Object.Destroy(_cs);
        }
        
    }
}