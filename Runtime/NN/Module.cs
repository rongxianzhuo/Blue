using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Blue.Graph;
using UnityEngine;

namespace Blue.Runtime.NN
{
    
    public abstract class Module : IDisposable
    {

        private List<ComputationalNode> _parameters;

        public IReadOnlyList<ComputationalNode> Parameters
        {
            get
            {
                if (_parameters == null)
                {
                    _parameters = new List<ComputationalNode>();
                    GetParameters(_parameters);
                }
                return _parameters;
            }
        }

        public abstract ComputationalNode CreateGraph(params ComputationalNode[] input);

        public ComputationalGraph Forward(params ComputationalNode[] input)
        {
            return new ComputationalGraph(CreateGraph(input));
        }

        private void GetParameters(ICollection<ComputationalNode> list)
        {
            var moduleType = typeof(Module);
            var nodeType = typeof(ComputationalNode);
            foreach (var field in GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            {
                if (field.FieldType.IsSubclassOf(moduleType)) ((Module)field.GetValue(this)).GetParameters(list);
                if (field.FieldType == nodeType)
                {
                    var node = (ComputationalNode)field.GetValue(this);
                    if (node.IsParameter) list.Add(node);
                }
            }
        }
            
        public void LoadFromFile(string dirPath)
        {
            for (var i = 0; i < Parameters.Count; i++)
            {
                var path = Path.Combine(dirPath, $"{i}.bytes");
                if (File.Exists(path))
                {
                    _parameters[i].LoadFromFile(path);
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {path}");
                }
            }
        }

        public void SaveToFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            for (var i = 0; i < Parameters.Count; i++)
            {
                var path = Path.Combine(dirPath, $"{i}.bytes");
                using var stream = File.OpenWrite(path);
                _parameters[i].SaveToStream(stream);
                stream.Close();
            }
        }

        public void Dispose()
        {
            var moduleType = typeof(Module);
            var nodeType = typeof(ComputationalNode);
            foreach (var field in GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
            {
                if (field.FieldType.IsSubclassOf(moduleType)) ((Module)field.GetValue(this)).Dispose();
                if (field.FieldType == nodeType) ((ComputationalNode)field.GetValue(this)).Dispose();
            }
        }
    }
}