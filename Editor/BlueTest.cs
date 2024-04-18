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

        private const string Py = @"
def save_tensor_list(save_path, *tensor_list):
    with open(save_path, 'wb') as file:
        for tensor in tensor_list:
            for f in tensor.reshape(-1).detach().numpy():
                file.write(struct.pack('f', float(f)))
";

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
            CheckFloatValueSimilar(t1.Sync(), packer.UnpackSingleArray(t1.FlattenSize));
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
            
            using var a = new ComputationalNode(true, 32, 8);
            a.LoadFromStream(stream);
            using var b = a.Tanh().Forward();
            using var loss = new MseLoss(b);
            loss.Target.LoadFromStream(stream);
            loss.Backward();
            
            CheckFloatValueSimilar(b, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("Tanh Pass");
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
        public static void Mul()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Mul.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            using var b = new ComputationalNode(true, 32, 8);
            var c = a * b;
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
            Debug.Log("Mul Pass");
        }

        [Test]
        public static void MatMul()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/MatMul.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32);
            using var b = new ComputationalNode(true, 32, 16);
            var c = a.MatMul(b);
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
            Debug.Log("MatMul Pass");
        }

        [Test]
        public static void Transpose()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Transpose.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            var b = a.Transpose();
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
        public static void Power()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Power.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 2, 3);
            var b = a.Power(0.5f);
            using var graph = b.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
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
            CheckFloatValueSimilar(a.Gradient.Sync(), 0f, -0.0690356f, -0.1220085f, -0.1666667f, 0f, 0f);
            Debug.Log("MaskedFill Pass");
        }

        [Test]
        public static void Softmax()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Softmax.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            var b = a.Softmax(1);
            using var graph = b.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            Debug.Log("Softmax Pass");
        }

        [Test]
        public static void LayerNorm()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/LayerNorm.bytes";
            using var stream = File.OpenRead(path);

            using var ln = new LayerNorm(5, 8);
            using var a = new ComputationalNode(true, 3, 5, 8);
            var b = ln.Build(a);
            using var graph = b.Graph();
            using var loss = new MseLoss(graph.Output);
            a.LoadFromStream(stream);
            loss.Target.LoadFromStream(stream);
            graph.Forward();
            loss.Backward();
            
            CheckFloatValueSimilar(graph.Output, stream);
            CheckFloatValueSimilar(a.Gradient, stream);
            CheckFloatValueSimilar(ln.Weight.Gradient, stream);
            CheckFloatValueSimilar(ln.Bias.Gradient, stream);
            Debug.Log("LayerNorm Pass");
        }

        [Test]
        public static void Add()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Add.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            using var b = new ComputationalNode(true, 1, 32, 8);
            var c = a + b;
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
            Debug.Log("Add Pass");
        }

        [Test]
        public static void Res()
        {
            var path = Application.dataPath + "/Blue/Editor/TestData/Res.bytes";
            using var stream = File.OpenRead(path);
            
            using var a = new ComputationalNode(true, 3, 32, 8);
            using var b = new ComputationalNode(true, 3, 32, 8);
            var c = a.Softmax(1) + (a * b).Sigmoid() + (a + b).ReLU() + b.Softmax(2);
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