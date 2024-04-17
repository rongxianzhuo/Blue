using System;
using System.Collections.Generic;
using System.IO;
using Blue.Data;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor : IDisposable
    {

        public readonly bool IsView;
        public readonly int[] Size;
        public readonly int FlattenSize;
        private readonly ComputeBuffer _buffer;

        private float[] _syncArray;

        public Tensor(Tensor origin, params int[] size)
        {
            IsView = true;
            Size = size;
            var totalSize = 1;
            foreach (var i in size)
            {
                totalSize *= i;
            }
            FlattenSize = totalSize;
            _buffer = origin._buffer;
        }

        public Tensor(params int[] size)
        {
            Size = size;
            var totalSize = 1;
            foreach (var i in size)
            {
                totalSize *= i;
            }
            FlattenSize = totalSize;
            _buffer = new ComputeBuffer(totalSize, sizeof(float));
            Op.Clear(this, 0).Dispatch().Dispose();
        }

        public Tensor(List<float> list)
        {
            Size = new []{list.Count};
            FlattenSize = list.Count;
            _buffer = new ComputeBuffer(list.Count, sizeof(float));
            _buffer.SetData(list);
        }

        public float Max(out int index)
        {
            Sync();
            index = 0;
            var maxValue = _syncArray[0];
            for (var i = 1; i < _syncArray.Length; i++)
            {
                if (_syncArray[i] < maxValue) continue;
                index = i;
                maxValue = _syncArray[i];
            }
            return maxValue;
        }

        public void LoadFromStream(Stream stream)
        {
            SetData(new MessagePacker(stream).UnpackSingleArray(FlattenSize));
        }

        public void LoadFromFile(string path)
        {
            using var stream = File.OpenRead(path);
            LoadFromStream(stream);
            stream.Close();
        }

        public void SaveToStream(Stream stream)
        {
            Sync();
            new MessagePacker(stream).Pack(_syncArray);
        }

        public void SaveToFile(string path)
        {
            using var stream = File.OpenWrite(path);
            SaveToStream(stream);
            stream.Close();
        }

        public void SetData(params float[] data)
        {
            _buffer.SetData(data);
        }

        public void SetData(IEnumerable<float[]> data)
        {
            InternalSync(false);
            var i = 0;
            foreach (var array in data)
            {
                foreach (var f in array)
                {
                    _syncArray[i++] = f;
                }
            }
            _buffer.SetData(_syncArray);
        }

        public void SetData(Action<float[]> setter)
        {
            setter(InternalSync(false));
            _buffer.SetData(_syncArray);
        }

        public void GetData(float[] array)
        {
            _buffer.GetData(array);
        }

        public void SetToShader(ComputeShader cs, int kernel, string propertyName)
        {
            cs.SetBuffer(kernel, propertyName, _buffer);
        }

        public IReadOnlyList<float> Sync() => InternalSync(true);

        internal float[] InternalSync(bool getData)
        {
            if (_syncArray == null || _syncArray.Length != FlattenSize)
            {
                _syncArray = new float[FlattenSize];
            }
            if (getData) _buffer.GetData(_syncArray);
            return _syncArray;
        }

        public virtual void Dispose()
        {
            if (!IsView) _buffer.Release();
        }

        public void Print(char sep=',')
        {
            var array = Sync();
            if (Size.Length > 1)
            {
                var builder = new System.Text.StringBuilder();
                var len = FlattenSize / Size[0];
                for (var i = 0; i < Size[0]; i++)
                {
                    builder.Append(array[len * i]);
                    for (var j = 1; j < len; j++)
                    {
                        builder.Append(',');
                        builder.Append(array[len * i + j]);
                    }
                    if (i < Size[0] - 1) builder.Append('\n');
                }
                Debug.Log(builder);
            }
            else
            {
                Debug.Log(string.Join(sep, array));
            }
        }
    }
}