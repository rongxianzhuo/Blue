using System;
using System.Collections.Generic;
using System.IO;
using Blue.Graph;
using UnityEngine;

namespace Blue.Runtime.NN
{
    
    public abstract class Module : IDisposable
    {

        private readonly List<Module> _subModules = new List<Module>();

        private readonly List<ComputationalNode> _parameters = new List<ComputationalNode>();


        public abstract ComputationalNode Forward(params ComputationalNode[] input);

        protected void RegisterModule(Module module)
        {
            _subModules.Add(module);
        }

        protected ComputationalNode CreateParameter(params int[] size)
        {
            var node = new ComputationalNode(true, size);
            _parameters.Add(node);
            return node;
        }

        public List<ComputationalNode> GetAllParameters(List<ComputationalNode> list=null)
        {
            list ??= new List<ComputationalNode>();
            list.AddRange(_parameters);
            foreach (var module in _subModules)
            {
                module.GetAllParameters(list);
            }
            return list;
        }
            
        public void LoadFromFile(string dirPath)
        {
            var allParameter = GetAllParameters();
            for (var i = 0; i < allParameter.Count; i++)
            {
                var path = Path.Combine(dirPath, $"{i}.bytes");
                if (File.Exists(path))
                {
                    allParameter[i].LoadFromFile(path);
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {path}");
                }
            }
        }

        public void SaveToFile(string dirPath)
        {
            var allParameter = GetAllParameters();
            Directory.CreateDirectory(dirPath);
            for (var i = 0; i < allParameter.Count; i++)
            {
                var path = Path.Combine(dirPath, $"{i}.bytes");
                allParameter[i].SaveToFile(path);
            }
        }

        public void Dispose()
        {
            foreach (var node in _parameters)
            {
                node.Dispose();
            }
            _parameters.Clear();
            foreach (var module in _subModules)
            {
                module.Dispose();
            }
            _subModules.Clear();
        }
    }
}