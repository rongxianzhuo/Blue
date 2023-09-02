using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Blue.Core;
using UnityEngine;
using Blue.Kit;

namespace Blue.Graph
{
    public class TensorNode : IGraphNode
    {

        public readonly Tensor TotalGradient;
        public readonly string Name;

        private readonly Tensor _output;
        private readonly Tensor _gradient;

        public bool IsParameter => TotalGradient != null;

        public TensorNode(string name, int size, bool isParam)
        {
            Name = name;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(size);
            _gradient = new Tensor(size);
        }

        public TensorNode(string name, bool isParam, List<float> data)
        {
            Name = name;
            var size = data.Count;
            TotalGradient = isParam ? new Tensor(size) : null;
            _output = new Tensor(data);
            _gradient = new Tensor(size);
        }
        
        public Tensor GetOutput()
        {
            return _output;
        }

        public Tensor GetGradient()
        {
            return _gradient;
        }

        public void Forward()
        {
        }

        public void Backward()
        {
            if (TotalGradient == null) return;
            Op.Increment(TotalGradient, _gradient);
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