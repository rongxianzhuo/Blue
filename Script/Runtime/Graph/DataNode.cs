using System;
using Blue.Operates;
using UnityEngine;

namespace Blue.Graph
{
    public class DataNode : IGraphNode
    {

        [Serializable]
        public class SerializedObject
        {
            public float[] data;
        }

        public readonly ComputeBuffer TotalGradient;
        public readonly string Name;

        private readonly ComputeBuffer _output;
        private readonly ComputeBuffer _gradient;

        public bool IsParameter => TotalGradient != null;

        public DataNode(string name, int size, bool isParam)
        {
            Name = name;
            TotalGradient = isParam ? new ComputeBuffer(size, 4) : null;
            _output = new ComputeBuffer(size, 4);
            _gradient = new ComputeBuffer(size, 4);
        }

        public void LoadFromText(string text)
        {
            var so = JsonUtility.FromJson<SerializedObject>(text);
            _output.SetData(so.data);
        }

        public string SaveAsText()
        {
            var so = new SerializedObject
            {
                data = new float[_output.count]
            };
            _output.GetData(so.data);
            return JsonUtility.ToJson(so);
        }
        
        public ComputeBuffer GetOutput()
        {
            return _output;
        }

        public ComputeBuffer GetGradient()
        {
            return _gradient;
        }

        public void Calculate()
        {
        }

        public void GradientPropagation()
        {
            if (TotalGradient != null) AddOperate.Calculate(TotalGradient, _gradient, 1, 0);
        }

        public void Destroy()
        {
            _output.Release();
            _gradient.Release();
            TotalGradient?.Release();
        }

        public void ForeachInputNode(Action<IGraphNode> action)
        {
        }
    }
}