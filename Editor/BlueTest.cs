using System;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Data;
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
            if (float.IsNaN(f1)) throw new Exception("NaN");
            if (float.IsNaN(f2)) throw new Exception("NaN");
            if (Mathf.Abs(f1 - f2) > 0.00001f) throw new Exception($"Not similar: {f1} {f2} {Mathf.Abs(f1 - f2)}");
        }

        private static void CheckFloatValueSimilar(IReadOnlyList<float> f1, params float[] f2)
        {
            if (f1.Count != f2.Length) throw new Exception("Not similar");
            for (var i = 0; i < f1.Count; i++)
            {
                CheckFloatValueSimilar(f1[i], f2[i]);
            }
        }

        private static void CheckFloatValueSimilar(Tensor t1, Stream stream)
        {
            var packer = new MessagePacker(stream);
            CheckFloatValueSimilar(t1.Sync<float>(), packer.UnpackSingleArray(t1.FlattenSize));
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
            var path = Application.dataPath + "/Blue/Editor/TestData/Tanh.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            a.LoadFromStream(stream);
            using var b = a.Transpose(0, 2);
            using var c = b.Tanh().Forward();
            using var loss = new MseLoss(c);
            loss.Target.LoadFromStream(stream);
            loss.Backward();
            
            CheckFloatValueSimilar(c, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("Tanh Pass");
        }

        [Test]
        public static void ReLU()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/ReLU.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            a.LoadFromStream(stream);
            using var b = a.Transpose(0, 2);
            using var c = b.ReLU().Forward();
            using var loss = new MseLoss(c);
            loss.Target.LoadFromStream(stream);
            loss.Backward();
            
            CheckFloatValueSimilar(c, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("ReLU Pass");
        }

        [Test]
        public static void Sigmoid()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Sigmoid.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            a.LoadFromStream(stream);
            using var b = a.Transpose(0, 2);
            using var c = b.Sigmoid().Forward();
            using var loss = new MseLoss(c);
            loss.Target.LoadFromStream(stream);
            loss.Backward();
            
            CheckFloatValueSimilar(c, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("Sigmoid Pass");
        }

        [Test]
        public static void AddInPlace()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/AddInPlace.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            a.LoadFromStream(stream);
            using var b = new ComputationalNode(true, 32, 8);
            b.LoadFromStream(stream);
            a.AddInPlace(b).Forward();
            using var loss = new MseLoss(a);
            loss.Target.LoadFromStream(stream);
            loss.Backward();
            
            CheckFloatValueSimilar(a, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(b.Gradient, stream);
            Debug.Log("AddInPlace Pass");
        }

        [Test]
        public static void MatMul1()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/MatMul1.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 32, 8);
            using var b = new ComputationalNode(true, 32, 8);
            var c = a.Transpose(0, 1);
            var d = c.MatMul(b);
            using var graph = d.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            b.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(b.Gradient, stream);
            Debug.Log("MatMul1 Pass");
        }

        [Test]
        public static void MatMul2()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/MatMul2.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 32, 8);
            using var b = new ComputationalNode(true, 32, 8);
            var c = b.Transpose(0, 1);
            var d = a.MatMul(c);
            using var graph = d.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            b.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(b.Gradient, stream);
            Debug.Log("MatMul2 Pass");
        }

        [Test]
        public static void Transpose()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Transpose.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            var b = a.Transpose(0, 2);
            using var graph = b.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("Transpose Pass");
        }

        [Test]
        public static void Add()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Add.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            a.LoadFromStream(stream);
            using var b = new ComputationalNode(true, 8, 32, 3);
            b.LoadFromStream(stream);
            var c = b.Transpose(0, 2);
            var d = a + c;
            using var graph = d.Graph();
            using var loss = new MseLoss(graph.Output);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(b.Gradient, stream);
            Debug.Log("Add Pass");
        }

        [Test]
        public static void Res()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Res.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            using var b = new ComputationalNode(true, 3, 32, 8);
            var c = (a * b).Sigmoid() + (a + b).ReLU();
            using var graph = c.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            b.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(b.Gradient, stream);
            Debug.Log("Res Pass");
        }
        
    }

}