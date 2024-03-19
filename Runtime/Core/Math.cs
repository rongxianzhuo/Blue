using UnityEngine;

namespace Blue.Core
{
    public static class Math
    {
        
        public static float RandN(float a, float v)
        {
            var u1 = Random.Range(0f, 1f);
            var u2 = Random.Range(0f, 1f);
            var n = Mathf.Sqrt(-2 * Mathf.Log(u1)) * Mathf.Cos(2 * Mathf.PI * u2);
            return n * v + a;
        }
    }
}