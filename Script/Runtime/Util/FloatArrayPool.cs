using System;
using System.Collections.Generic;
using UnityEngine;

namespace Blue.Util
{
    public class FloatArrayPool
    {

        public static FloatArrayPool Default = new FloatArrayPool();

        private static readonly HashSet<float[]> RecycledArray = new HashSet<float[]>();

        public float[] Get(int size)
        {
            foreach (var array in RecycledArray)
            {
                if (array.Length != size) continue;
                RecycledArray.Remove(array);
                return array;
            }

            var a = new float[size];
            RecycledArray.Add(a);
            return a;
        }

        public float[] Get(float[] buffer)
        {
            var array = Get(buffer.Length);
            Array.Copy(buffer, array, array.Length);
            return array;
        }

        public float[] Get(ComputeBuffer buffer)
        {
            var array = Get(buffer.count);
            buffer.GetData(array);
            return array;
        }

        public void Recycle(float[] array)
        {
            RecycledArray.Add(array);
        }
    }
}