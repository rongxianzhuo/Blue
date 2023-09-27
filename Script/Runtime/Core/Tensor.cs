using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor
    {

        private float[] _syncArray;
        private ComputeBuffer _buffer;

        public int[] Size { get; private set; }

        public int FlattenSize { get; private set; }

        public Tensor(params int[] size)
        {
            Resize(size);
        }

        public bool IsSize(params int[] size)
        {
            if (Size == null) return false;
            if (size.Length != Size.Length) return false;
            for (var i = 0; i < size.Length; i++)
            {
                if (size[i] != Size[i]) return false;
            }

            return true;
        }

        public Tensor(List<float> list)
        {
            Size = new []{list.Count};
            FlattenSize = list.Count;
            _buffer = new ComputeBuffer(list.Count, sizeof(float));
            _buffer.SetData(list);
        }

        public bool Resize(params int[] size)
        {
            return ResizeWithValue(0f, size);
        }

        public bool ResizeWithValue(float fillValue, params int[] size)
        {
            if (IsSize(size)) return false;
            _syncArray = null;
            if (_buffer != null) _buffer.Release();
            Size = size;
            var totalSize = 1;
            foreach (var i in size)
            {
                totalSize *= i;
            }
            FlattenSize = totalSize;
            _buffer = new ComputeBuffer(totalSize, sizeof(float));
            Op.Clear(this, fillValue).Dispatch().Destroy();
            return true;
        }

        public void LoadFromStream(Stream stream)
        {
            const int sizeOfFloat = sizeof(float);
            var bytes = new byte[sizeOfFloat];
            var array = new float[FlattenSize];
            for (var i = 0; i < FlattenSize; i++)
            {
                for (var j = 0; j < sizeOfFloat; j++)
                {
                    bytes[j] = (byte) stream.ReadByte();
                }

                array[i] = BitConverter.ToSingle(bytes);
            }
            SetData(array);
        }

        public void SaveToStream(Stream stream)
        {
            Sync();
            const int sizeOfFloat = sizeof(float);
            var bytes = new byte[sizeOfFloat];
            foreach (var f in _syncArray)
            {
                if (!BitConverter.TryWriteBytes(bytes, f))
                {
                    throw new Exception("Unknown error");
                }

                for (var i = 0; i < sizeOfFloat; i++)
                {
                    stream.WriteByte(bytes[i]);
                }
            }
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

        public void SetToShader(ComputeShader cs, int kernel, string propertyName)
        {
            cs.SetBuffer(kernel, propertyName, _buffer);
        }

        public IReadOnlyList<float> Sync()
        {
            if (_syncArray == null || _syncArray.Length != FlattenSize)
            {
                _syncArray = new float[FlattenSize];
            }
            _buffer.GetData(_syncArray);
            return _syncArray;
        }

        public Tensor Transpose()
        {
            var size = new int[Size.Length];
            for (var i = 0; i < size.Length; i++)
            {
                size[i] = Size[size.Length - i - 1];
            }
            var result = new Tensor(size);
            var op = Op.Transpose(this, result);
            op.Dispatch();
            op.Destroy();
            return result;
        }

    }
}