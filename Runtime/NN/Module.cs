using System;
using System.Collections.Generic;
using System.IO;
using Blue.Core;
using Blue.Graph;
using Blue.Kit;
using UnityEngine;

namespace Blue.NN
{
    
    public abstract class Module : IDisposable
    {

        private readonly List<Module> _subModules = new List<Module>();

        private readonly List<ComputationalNode> _parameters = new List<ComputationalNode>();


        public abstract ComputationalNode Build(params ComputationalNode[] input);

        protected T RegisterModule<T>(T module) where T : Module
        {
            _subModules.Add(module);
            return module;
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
            
        public void LoadFromFile(string path)
        {
            using var stream = File.OpenRead(path);
            LoadFromStream(stream);
            stream.Close();
        }
            
        public void LoadFromStream(Stream stream)
        {
            foreach (var t in GetAllParameters())
            {
                t.LoadFromStream(stream);
            }
            foreach (var t in _subModules)
            {
                t.LoadFromStream(stream);
            }
        }

        public void SaveToFile(string path)
        {
            using var stream = File.OpenWrite(path);
            SaveToStream(stream);
            stream.Close();
        }
            
        public void SaveToStream(Stream stream)
        {
            foreach (var t in GetAllParameters())
            {
                t.SaveToStream(stream);
            }
            foreach (var t in _subModules)
            {
                t.SaveToStream(stream);
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

        public OperateList Lerp(Module other, Tensor t, OperateList operateList=null)
        {
            operateList ??= new OperateList();
            for (var i = 0; i < _parameters.Count; i++)
            {
                operateList.Add(Op.Lerp(_parameters[i], other._parameters[i], t));
            }
            for (var i = 0; i < _subModules.Count; i++)
            {
                _subModules[i].Lerp(other._subModules[i], t, operateList);
            }
            return operateList;
        }

        public OperateList CopyParameter(Module other, OperateList list=null)
        {
            list ??= new OperateList();
            for (var i = 0; i < _parameters.Count; i++)
            {
                list.Add(Op.Copy(other._parameters[i], 0, 0
                    , _parameters[i], 0, 0
                    , _parameters[i].FlattenSize
                    , _parameters[i].FlattenSize));
                _parameters[i].SetData(other._parameters[i].InternalSync());
            }
            for (var i = 0; i < _subModules.Count; i++)
            {
                _subModules[i].CopyParameter(other._subModules[i], list);
            }
            return list;
        }
    }
}