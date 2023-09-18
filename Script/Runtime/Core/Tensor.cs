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

        [Serializable]
        public class FloatArrayData
        {
            public float[] data;
        }

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

        public void LoadFromJson(string json)
        {
            var data = JsonUtility.FromJson<FloatArrayData>(json);
            SetData(data.data);
        }

        public void LoadFromStream(Stream stream)
        {
            var binaryFormatter = new BinaryFormatter();
            var array = (float[])binaryFormatter.Deserialize(stream);
            SetData(array);
        }

        public void Resize(params int[] size)
        {
            if (IsSize(size)) return;
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
            Op.Clear(this, 0);
        }

        public void SaveToStream(Stream stream)
        {
            Sync();
            var binaryFormatter = new BinaryFormatter();
            binaryFormatter.Serialize(stream, _syncArray);
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

        public void SetToShader(ComputeShader cs, int kernel, string propertyName)
        {
            cs.SetBuffer(kernel, propertyName, _buffer);
        }

        public IReadOnlyList<float> Sync()
        {
            _syncArray ??= new float[FlattenSize];
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
            Op.Transpose(this, result);
            return result;
        }

    }
}