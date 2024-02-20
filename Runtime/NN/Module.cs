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

        private readonly List<ComputationalNode> _tempNodes = new List<ComputationalNode>();

        private readonly List<ComputationalNode> _parameters = new List<ComputationalNode>();


        public abstract ComputationalNode Forward(params ComputationalNode[] input);

        protected void RegisterModule(Module module)
        {
            _subModules.Add(module);
        }

        protected void RegisterParameter(ComputationalNode node)
        {
            _parameters.Add(node);
        }

        protected void RegisterTempNode(ComputationalNode node)
        {
            _tempNodes.Add(node);
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
            foreach (var node in _tempNodes)
            {
                node.Dispose();
            }
            _tempNodes.Clear();
            foreach (var module in _subModules)
            {
                module.Dispose();
            }
            _subModules.Clear();
        }
    }
}