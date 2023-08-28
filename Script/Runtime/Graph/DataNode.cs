using System;
using UnityEngine;


using Blue.Core;namespace Blue.Graph
{
    public class DataNode : IGraphNode
    {

        private static Operate _incrementOperate;

        private static Operate GetIncrementOperate() => _incrementOperate ??= new Operate("Common/Increment", "CSMain"
            , "r_buffer1", "rw_buffer1");

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

        public void Forward()
        {
        }

        public void Backward()
        {
            if (TotalGradient == null) return;
            
            GetIncrementOperate().CreateTask()
                .SetBuffer(_gradient)
                .SetBuffer(TotalGradient)
                .Dispatch(new Vector3Int(TotalGradient.count, 1, 1));
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