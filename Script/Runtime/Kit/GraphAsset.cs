using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

namespace Blue.Kit
{
    
    public class GraphAsset
    {

        [Serializable]
        public class NodeDisplayInfo
        {

            public Rect position;

        }

        public NodeDisplayInfo OutputNodeInfo { get; private set; }

        private readonly List<string> _nodeTree = new List<string>();

        private readonly Dictionary<string, string> _methods = new Dictionary<string, string>();

        private readonly Dictionary<string, object[]> _parameters = new Dictionary<string, object[]>();

        private readonly Dictionary<string, NodeDisplayInfo> _infos = new Dictionary<string, NodeDisplayInfo>();

        public string OutputNodeName => _nodeTree[_nodeTree.Count - 1];

        public void ForeachNode(Action<string, object[], NodeDisplayInfo> action)
        {
            foreach (var name in _nodeTree)
            {
                action(_methods[name], _parameters[name], _infos[name]);
            }
        }

        public void AddNode(string name, string method, object[] parameters, NodeDisplayInfo info)
        {
            _nodeTree.Add(name);
            _methods[name] = method;
            _parameters[name] = parameters;
            _infos[name] = info;
        }

        public void LoadFromStream(Stream stream)
        {
            var binaryFormatter = new BinaryFormatter();
            OutputNodeInfo = JsonUtility.FromJson<NodeDisplayInfo>((string)binaryFormatter.Deserialize(stream));
            while (stream.Position < stream.Length)
            {
                AddNode((string)binaryFormatter.Deserialize(stream)
                    ,  (string)binaryFormatter.Deserialize(stream)
                    ,  (object[])binaryFormatter.Deserialize(stream)
                    ,  JsonUtility.FromJson<NodeDisplayInfo>((string)binaryFormatter.Deserialize(stream)));
            }
        }

        public void SaveToStream(Stream stream, NodeDisplayInfo outputNodeInfo)
        {
            var binaryFormatter = new BinaryFormatter();
            binaryFormatter.Serialize(stream, JsonUtility.ToJson(outputNodeInfo));
            foreach (var name in _nodeTree)
            {
                binaryFormatter.Serialize(stream, name);
                binaryFormatter.Serialize(stream, _methods[name]);
                binaryFormatter.Serialize(stream, _parameters[name]);
                binaryFormatter.Serialize(stream, JsonUtility.ToJson(_infos[name]));
            }
        }

        public ModelBuilder CreateBuilder()
        {
            var modelBuilderType = typeof(ModelBuilder);
            var builder = new ModelBuilder();
            foreach (var name in _nodeTree)
            {
                modelBuilderType.GetMethod(_methods[name])?.Invoke(builder, _parameters[name]);
            }
            return builder;
        }

    }
}