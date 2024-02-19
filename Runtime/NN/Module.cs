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


        public abstract ComputationalNode CreateGraph(params ComputationalNode[] input);

        public ComputationalGraph Forward(params ComputationalNode[] input)
        {
            return new ComputationalGraph(CreateGraph(input));
        }

        protected void RegisterModule(Module module)
        {
            _subModules.Add(module);
        }

        protected void RegisterParameter(ComputationalNode node)
        {
            _parameters.Add(node);
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
                using var stream = File.OpenWrite(path);
                allParameter[i].SaveToStream(stream);
                stream.Close();
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