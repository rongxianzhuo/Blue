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

        private class Test : Attribute
        {
            
        }

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

        [MenuItem("Blue/TestAll")]
        public static void TestAll()
        {
            var p = new object[] { };
            foreach (var m in typeof(BlueTest).GetMethods())
            {
                var isTest = false;
                foreach (var o in m.GetCustomAttributes(false))
                {
                    if (o is Test)
                    {
                        isTest = true;
                        break;
                    }
                }
                if (isTest) m.Invoke(null, p);
            }
        }

        [Test]
        public static void Tanh()
        {
            using var a = new ComputationalNode(true, 1, 6);
            a.SetData(1.0f, -0.5f, 0.1f, -0.2f, 0.0f, -6.5f);
            using var b = Activation.Tanh.Build(a);
            b.Forward();
            CheckFloatValueSimilar(b.Sync(), 0.7616f, -0.4621f,  0.0997f, -0.1974f,  0.0000f, -1.0000f);
            b.Gradient.SetData(2f, 2f, 2f, 2f, 2f, 2f);
            b.Backward();
            CheckFloatValueSimilar(a.Gradient.Sync(), 8.3995e-01f, 1.5729e+00f, 1.9801e+00f, 1.9221e+00f, 2.0000e+00f, 1.8120e-05f);
            Debug.Log("Tanh Pass");
        }

        [Test]
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

        [Test]
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

        [Test]
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

        [Test]
        public static void Transpose()
        {
            using var a = new ComputationalNode(true, 2, 2, 3);
            a.SetData(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f);
            using var b = a.Transpose();
            b.Forward();
            CheckFloatValueSimilar(b.Sync(), 1f, 4f, 2f, 5f, 3f, 6f, 7f, 10f, 8f, 11f, 9f, 12f);
            using var loss = new MseLoss(b);
            loss.Backward();
            CheckFloatValueSimilar(a.Gradient.Sync(), 0.1667f, 0.3333f, 0.5f, 0.6667f, 0.8333f, 1f, 1.1667f, 1.3333f, 1.5f, 1.6667f, 1.8333f, 2f);
            Debug.Log("Transpose Pass");
        }

        [Test]
        public static void Power()
        {
            using var a = new ComputationalNode(true, 2, 3);
            a.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            using var b = a.Power(0.5f);
            b.Forward();
            using var loss = new MseLoss(b);
            loss.Target.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            loss.Backward();
            CheckFloatValueSimilar(a.Gradient.Sync(), 0f, -0.069f, -0.122f, -0.1667f, -0.206f, -0.2416f);
            Debug.Log("Power Pass");
        }

        [Test]
        public static void MaskedFill()
        {
            using var a = new ComputationalNode(true, 2, 3);
            a.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            using var b = a.Power(0.5f);
            b.Forward();
            using var mask = new ComputationalNode(false, 2, 3);
            mask.SetData(0f, 0f, 0f, 0f, 1f, 1f);
            using var c = b.MaskedFill(mask, float.NegativeInfinity);
            c.Forward();
            using var loss = new MseLoss(c);
            loss.Target.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            loss.Backward();
            CheckFloatValueSimilar(a.Gradient.Sync(), 0f, -0.069f, -0.122f, -0.1667f, 0f, 0f);
            Debug.Log("MaskedFill Pass");
        }

        [Test]
        public static void Softmax()
        {
            using var a = new ComputationalNode(true, 2, 3);
            a.SetData(1f, 2f, 3f, 4f, 5f, 7f);
            using var b = a.Softmax(1);
            b.Forward();
            CheckFloatValueSimilar(b.Sync(), 0.09f, 0.2447f, 0.6652f, 0.042f, 0.1142f, 0.8438f);
            using var loss = new MseLoss(b);
            loss.Target.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            loss.Backward();
            CheckFloatValueSimilar(a.Gradient.Sync(), 0.0347f, 0.0252f, -0.0599f, 0.0156f, 0.0072f, -0.0228f);
            
            using var a1 = new ComputationalNode(true, 2, 3);
            a1.SetData(1f, 2f, 3f, 4f, 5f, 7f);
            using var b1 = a1.Softmax(0);
            b1.Forward();
            using var loss1 = new MseLoss(b1);
            loss1.Target.SetData(1f, 2f, 3f, 4f, 5f, 6f);
            loss1.Backward();
            CheckFloatValueSimilar(a1.Gradient.Sync(), 0.0315f, 0.0315f, 0.012f, -0.0315f, -0.0315f, -0.012f);
            Debug.Log("Softmax Pass");
        }
        
    }

}