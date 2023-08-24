using System.Collections.Generic;

namespace Blue.Util
{
    public class SmoothHelper
    {
        
        private readonly double[] _buffer;

        private int _i;

        public SmoothHelper(int capacity)
        {
            _buffer = new double[capacity];
        }

        public void Add(double f)
        {
            _buffer[_i++] = f;
            _i %= _buffer.Length;
        }

        public float Mean()
        {
            double sum = 0;
            foreach (var f in _buffer)
            {
                sum += f;
            }

            return (float) (sum / _buffer.Length);
        }

    }
}