using UnityEngine;

namespace Blue.Util
{
    public static class ComputeBufferExpansion
    {

        public static int MaxIndex(this ComputeBuffer buffer)
        {
            var array = FloatArrayPool.Default.Get(buffer);
            var max = array[0];
            var index = 0;
            for (var i = 1; i < array.Length; i++)
            {
                if (array[i] <= max) continue;
                max = array[i];
                index = i;
            }
            FloatArrayPool.Default.Recycle(array);
            return index;
        }

        public static float Sum(this ComputeBuffer buffer)
        {
            var result = 0f;
            var array = FloatArrayPool.Default.Get(buffer);
            foreach (var f in array)
            {
                result += f;
            }
            FloatArrayPool.Default.Recycle(array);
            return result;
        }

        public static float SquareSum(this ComputeBuffer buffer)
        {
            var result = 0f;
            var array = FloatArrayPool.Default.Get(buffer);
            foreach (var f in array)
            {
                result += f * f;
            }
            FloatArrayPool.Default.Recycle(array);
            return result;
        }

        public static void Print(this ComputeBuffer buffer, string tag="")
        {
            var array = FloatArrayPool.Default.Get(buffer);
            var result = string.Join(',', array);
            FloatArrayPool.Default.Recycle(array);
            Debug.Log($"{tag}: {result}");
        }
        
    }
}