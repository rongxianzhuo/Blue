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

        public IReadOnlyList<float> Sync()
        {
            _syncArray ??= new float[Size];
            _buffer.GetData(_syncArray);
            return _syncArray;
        }

    }
}