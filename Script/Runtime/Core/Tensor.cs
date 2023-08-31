using System.Collections.Generic;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor
    {

        public readonly int Size;

        private readonly ComputeBuffer _buffer;

        private float[] _syncArray;

        public Tensor(int size)
        {
            Size = size;
            _buffer = new ComputeBuffer(size, sizeof(float));
            Op.Clear(this, 0);
        }

        public Tensor(List<float> list)
        {
            Size = list.Count;
            _buffer = new ComputeBuffer(list.Count, sizeof(float));
            _buffer.SetData(list);
        }

        public void Release()
        {
            _buffer.Release();
        }

        public void SetData(float[] data)
        {
            Sync();
            _buffer.SetData(data);
        }

        public void GetData(float[] array)
        {
            _buffer.GetData(array);
        }

        public void SetToShader(ComputeShader cs, int kernel, int propertyId)
        {
            cs.SetBuffer(kernel, propertyId, _buffer);
        }

        public IReadOnlyList<float> Sync()
        {
            _syncArray ??= new float[Size];
            _buffer.GetData(_syncArray);
            return _syncArray;
        }

    }
}