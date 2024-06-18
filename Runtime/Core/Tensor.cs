using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Blue.Data;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    
    public class Tensor : IDisposable
    {
        
        public readonly bool IsContinuous = true;
        public readonly int[] Size;
        public readonly int FlattenSize;

        private readonly Tensor _originTensor;
        private readonly int[] _strideInMemory;
        private readonly ComputeBuffer _buffer;
        private readonly HashSet<ComputeBuffer> _strideBufferPool = new HashSet<ComputeBuffer>();

        private Array _syncArray;
        
        public bool IsView => _originTensor != null;
        
        public IReadOnlyList<int> StrideInMemory => _strideInMemory;

        public Tensor(IReadOnlyList<int> size
            , Tensor origin=null
            , IReadOnlyList<int> strideList=null
            , int stride=4)
        {
            _originTensor = origin;
            Size = size.ToArray();
            FlattenSize = 1;
            foreach (var i in size) FlattenSize *= i;
            _strideInMemory = strideList == null ? CalculateStride(size) : strideList.ToArray();
            for (var i = 1; i < _strideInMemory.Length; i++)
            {
                if (_strideInMemory[i] <= _strideInMemory[i - 1]) continue;
                IsContinuous = false;
                break;
            }
            if (origin == null)
            {
                _buffer = new ComputeBuffer(FlattenSize, stride);
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
            if (IsContinuous)
            {
                _buffer.SetData(data);
            }
            else
            {
                var array = InternalSync<T>(false);
                var stride = CalculateStride(Size);
                for (var i = 0; i < array.Length; i++)
                {
                    var j = i;
                    var k = 0;
                    for (var l = 0; l < Size.Length; l++)
                    {
                        var d = j / stride[l];
                        k += d * _strideInMemory[l];
                        j -= d * stride[l];
                    }
                    array[k] = data[i];
                }
                _buffer.SetData(array);
            }
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
            SetData(syncArray);
        }

        public void SetData<T>(Action<T[]> setter)
        {
            setter(InternalSync<T>(false));
            SetData((T[])_syncArray);
        }

        public void GetData(float[] array)
        {
            _buffer.GetData(array);
        }

        public void SetToShader(ComputeShader cs, int kernel, string tensorName, int[] strideOrder)
        {
            if (strideOrder == null)
            {
                strideOrder = new int[Size.Length];
                for (var i = 0; i < Size.Length; i++)
                {
                    strideOrder[i] = i;
                }
            }
            cs.SetBuffer(kernel, $"{tensorName}_stride", CreateStrideBuffer(strideOrder));
            cs.SetBuffer(kernel, tensorName, _buffer);
            cs.SetInt($"{tensorName}_dim_size", Size.Length);
        }

        public IReadOnlyList<T> Sync<T>() => InternalSync<T>(true);

        private T[] InternalSync<T>(bool getData)
        {
            _syncArray ??= new T[FlattenSize];
            var array = (T[])_syncArray;
            if (!getData) return array;
            _buffer.GetData(array);
            if (IsContinuous) return array;
            var result = new T[array.Length];
            var stride = CalculateStride(Size);
            for (var i = 0; i < result.Length; i++)
            {
                var j = i;
                var k = 0;
                for (var l = 0; l < Size.Length; l++)
                {
                    var d = j / stride[l];
                    k += d * _strideInMemory[l];
                    j -= d * stride[l];
                }
                result[i] = array[k];
            }
            return result;
        }

        public virtual void Dispose()
        {
            if (!IsView) _buffer.Release();
            foreach (var buffer in _strideBufferPool) buffer.Dispose();
            _strideBufferPool.Clear();
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

        private ComputeBuffer CreateStrideBuffer(IReadOnlyList<int> order)
        {
            var stride = new int[order.Count];
            for (var i = 0; i < order.Count; i++)
            {
                stride[i] = _strideInMemory[order[i]];
            }
            var buffer = new ComputeBuffer(order.Count, sizeof(int));
            buffer.SetData(stride);
            _strideBufferPool.Add(buffer);
            return buffer;
        }
        
        private static int[] CalculateStride(IReadOnlyList<int> shape)
        {
            var stride = new int[shape.Count];
            var j = 1;
            for (var i = shape.Count - 1; i >= 0; i--)
            {
                stride[i] = j;
                j *= shape[i];
            }
            return stride;
        }

        public int[] CalculateStrideOrder()
        {
            var stride = _strideInMemory;
            var list = new List<int>();
            for (var i = 0; i < stride.Length; i++)
            {
                var max = 0;
                var maxIndex = -1;
                for (var j = 0; j < stride.Length; j++)
                {
                    if (max > stride[j]) continue;
                    if (list.Contains(j)) continue;
                    max = stride[j];
                    maxIndex = j;
                }
                list.Add(maxIndex);
            }
            return list.ToArray();
        }

    }
}