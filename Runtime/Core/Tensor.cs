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

        private Array _syncArray;

        public Tensor(params int[] size) : this(null, size)
        {
        }

        public Tensor(Tensor origin, params int[] size)
        {
            Size = size;
            var totalSize = 1;
            foreach (var i in size)
            {
                totalSize *= i;
            }
            FlattenSize = totalSize;
            IsView = origin != null;
            if (origin == null)
            {
                _buffer = new ComputeBuffer(totalSize, sizeof(float));
                Op.Clear(this, 0).Dispatch().Dispose();
            }
            else
            {
                _buffer = origin._buffer;
            }
        }

        public float Max(out int index)
        {
            var syncArray = Sync<float>();
            index = 0;
            var maxValue = syncArray[0];
            for (var i = 1; i < syncArray.Count; i++)
            {
                if (syncArray[i] < maxValue) continue;
                index = i;
                maxValue = syncArray[i];
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
            new MessagePacker(stream).Pack(InternalSync<float>(true));
        }

        public void SaveToFile(string path)
        {
            using var stream = File.OpenWrite(path);
            SaveToStream(stream);
            stream.Close();
        }

        public void SetData(Tensor other)
        {
            other._buffer.GetData(_syncArray);
            _buffer.SetData(_syncArray);
        }

        public void SetData<T>(T[] data)
        {
            _buffer.SetData(data);
        }

        public void SetData<T>(IEnumerable<T[]> data)
        {
            var syncArray = InternalSync<T>(false);
            var i = 0;
            foreach (var array in data)
            {
                foreach (var f in array)
                {
                    syncArray[i++] = f;
                }
            }
            _buffer.SetData(syncArray);
        }

        public void SetData<T>(Action<T[]> setter)
        {
            setter(InternalSync<T>(false));
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

        public IReadOnlyList<T> Sync<T>() => InternalSync<T>(true);

        internal T[] InternalSync<T>(bool getData)
        {
            _syncArray ??= new T[FlattenSize];
            if (getData) _buffer.GetData(_syncArray);
            return (T[])_syncArray;
        }

        public virtual void Dispose()
        {
            if (!IsView) _buffer.Release();
        }

        public void Print(char sep=',')
        {
            var array = InternalSync<float>(true);
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