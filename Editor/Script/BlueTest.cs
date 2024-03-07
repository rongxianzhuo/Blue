using System;
using System.Collections.Generic;
using Blue.Graph;
using Blue.Runtime.NN;
using UnityEditor;
using UnityEngine;


namespace Blue.Editor
{

    public static class BlueTest
    {

        private static void CheckFloatValueSimilar(float f1, float f2)
        {
            if (Mathf.Abs(f1 - f2) > 0.0001f) throw new Exception($"Not similar: {Mathf.Abs(f1 - f2)}");
        }

        private static void CheckFloatValueSimilar(IReadOnlyList<float> f1, params float[] f2)
        {
            if (f1.Count != f2.Length) throw new Exception("Not similar");
            for (var i = 0; i < f1.Count; i++)
            {
                CheckFloatValueSimilar(f1[i], f2[i]);
            }
        }

        [MenuItem("Blue/Test/AdditionAssignment")]
        public static void AdditionAssignment()
        {
            using var a = new ComputationalNode(true, 3, 2);
            a.SetData(0.1f, 0.5f, 0.8f, 0.25f, 0.36f, 0.89f);
            using var b = new ComputationalNode(true, 1, 2);
            b.SetData(0.13f, 0.25f);
            a.AdditionAssignment(b);
            a.Forward();
            using var loss = new MseLoss(a);
            loss.Target.SetData(0.35f, -0.25f, 0.8f, 0.51f, -0.36f, 0.29f);
            loss.Backward();
            CheckFloatValueSimilar(a.Sync(), 0.23f, 0.75f, 0.93f, 0.5f, 0.49f, 1.14f);
            CheckFloatValueSimilar(loss.Value, 0.4127f);
            CheckFloatValueSimilar(a.Gradient.Sync(), -0.04f, 0.3333f, 0.0433f, -0.0033f, 0.2833f, 0.2833f);
            CheckFloatValueSimilar(b.Gradient.Sync(), 0.2867f, 0.6133f);
            Debug.Log("AdditionAssignment Pass");
        }
        
    }

}