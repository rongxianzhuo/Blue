using System.Collections.Generic;

namespace Blue.Util
{
    public class SmoothHelper
    {
        
        private readonly float[] _buffer;

        private int _i;

        public SmoothHelper(int capacity)
        {
            _buffer = new float[capacity];
        }

        public void Add(float f)
        {
            _buffer[_i++] = f;
            _i %= _buffer.Length;
        }

        public float Mean()
        {
            var sum = 0f;
            foreach (var f in _buffer)
            {
                sum += f;
            }

            return sum / _buffer.Length;
        }

    }
}