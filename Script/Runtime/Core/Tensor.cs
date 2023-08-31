using System.Collections.Generic;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor
    {

        public readonly int Size;

        internal readonly ComputeBuffer Buffer;

        public bool IsValid { get; private set; }

        public Tensor(int size)
        {
            Size = size;
            Buffer = new ComputeBuffer(size, sizeof(float));
            IsValid = true;
        }

        public Tensor(List<float> list)
        {
            Size = list.Count;
            Buffer = new ComputeBuffer(list.Count, sizeof(float));
            Buffer.SetData(list);
            IsValid = true;
        }

        public void Release()
        {
            Buffer.Release();
            IsValid = false;
        }

        public void GetData(float[] array)
        {
            if (IsValid) Buffer.GetData(array);
        }

    }
}