using System;
using System.Collections.Generic;
using Blue.Graph;
using Blue.NN;
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

        [MenuItem("Blue/Test/Mul")]
        public static void Mul()
        {
            using var a = new ComputationalNode(true, 3, 2);
            a.SetData(0.1f, 0.5f, 0.8f, 0.25f, 0.36f, 0.89f);
            using var b = new ComputationalNode(true, 1, 2);
            b.SetData(0.13f, 0.25f);
            using var c = a * b;
            c.Forward();
            using var loss = new MseLoss(c);
            loss.Target.SetData(0.35f, -0.25f, 0.8f, 0.51f, -0.36f, 0.29f);
            loss.Backward();
            CheckFloatValueSimilar(c.Sync(), 0.013f, 0.125f, 0.104f, 0.0625f, 0.0468f, 0.2225f);
            CheckFloatValueSimilar(loss.Value, 0.1848f);
            CheckFloatValueSimilar(a.Gradient.Sync(), -0.0146f, 0.0312f, -0.0302f, -0.0373f, 0.0176f, -0.0056f);
            CheckFloatValueSimilar(b.Gradient.Sync(), -0.148f, 0.0052f);
            Debug.Log("Mul Pass");
        }

        [MenuItem("Blue/Test/MatMul")]
        public static void MatMul()
        {
            using var a = new ComputationalNode(true, 3, 2);
            a.SetData(0.1f, 0.5f, 0.8f, 0.25f, 0.36f, 0.89f);
            using var b = new ComputationalNode(true, 2, 3);
            b.SetData(0.1f, 0.5f, 0.8f, 0.25f, 0.36f, 0.89f);
            using var c = a.MatMul(b);
            c.Forward();
            using var loss = new MseLoss(c);
            loss.Target.SetData(1.35f, -2.25f, 1.8f, 2.51f, -1.36f, 2.29f, -1.1f, -3.3f, 0.6f);
            loss.Backward();
            CheckFloatValueSimilar(c.Sync(), 0.135f, 0.23f, 0.525f, 0.1425f, 0.49f, 0.8625f, 0.2585f, 0.5004f, 1.0801f);
            CheckFloatValueSimilar(loss.Value, 4.093f);
            CheckFloatValueSimilar(a.Gradient.Sync(), 0.0219f, -0.1213f, -0.1008f, -0.2659f, 0.5378f, 0.4745f);
            CheckFloatValueSimilar(b.Gradient.Sync(), -0.3392f, 0.688f, -0.2437f, 0.0022f, 1.13f, -0.126f);
            Debug.Log("MatMul Pass");
        }
        
    }

}