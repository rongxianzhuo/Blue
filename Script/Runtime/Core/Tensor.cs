using System.Collections.Generic;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor
    {

        public readonly int Size;

        private readonly ComputeBuffer _buffer;

        public bool IsValid { get; private set; }

        private float[] _syncArray;

        public Tensor(int size)
        {
            Size = size;
            _buffer = new ComputeBuffer(size, sizeof(float));
            IsValid = true;
        }

        public Tensor(List<float> list)
        {
            Size = list.Count;
            _buffer = new ComputeBuffer(list.Count, sizeof(float));
            _buffer.SetData(list);
            IsValid = true;
        }

        public void Release()
        {
            _buffer.Release();
            IsValid = false;
        }

        public void SetData(float[] data)
        {
            if (!IsValid) return;
            _syncArray ??= new float[Size];
            _buffer.GetData(_syncArray);
            _buffer.SetData(data);
        }

        public void GetData(float[] array)
        {
            if (IsValid) _buffer.GetData(array);
        }

        public void SetToShader(ComputeShader cs, int kernel, int propertyId)
        {
            cs.SetBuffer(kernel, propertyId, _buffer);
        }

    }
}