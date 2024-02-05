using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Blue.Graph;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    public class Model : IDisposable
    {

        public readonly ComputationalNode Output;

        private readonly ComputationalNode[] _inputNodes;
        private readonly List<ComputationalNode> _parameterNodes = new List<ComputationalNode>();
        private readonly List<Operate> _clearGradientOps = new List<Operate>();
        private readonly List<HashSet<ComputationalNode>> _nodeLayer = new List<HashSet<ComputationalNode>>();

        public IReadOnlyCollection<ComputationalNode> ParameterNodes => _parameterNodes;

        public Model(ComputationalNode outputNode, params ComputationalNode[] inputNodes)
        {
            Output = outputNode;
            _inputNodes = inputNodes;
            foreach (var node in outputNode.ReadOnlyInputNodes)
            {
                AddNode(node, outputNode);
            }
            for (var i = _nodeLayer.Count - 1; i >= 0; i--)
            {
                foreach (var node in _nodeLayer[i])
                {
                    if (node is ComputationalNode dataNode && dataNode.IsParameter) _parameterNodes.Add(dataNode);
                }
            }

            foreach (var node in _parameterNodes)
            {
                _clearGradientOps.Add(Op.Clear(node.TotalGradient, 0f));
            }
        }

        public void LoadParameterFile(string dirPath)
        {
            foreach (var node in _parameterNodes)
            {
                if (File.Exists(Path.Combine(dirPath, $"{node.Id}.bytes")))
                {
                    using var stream = File.OpenRead(Path.Combine(dirPath, $"{node.Id}.bytes"));
                    node.Output.LoadFromStream(stream);
                    stream.Close();
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {node.Id}");
                }
            }
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            foreach (var node in _parameterNodes)
            {
                using var stream = File.OpenWrite(Path.Combine(dirPath, $"{node.Id}.bytes"));
                node.Output.SaveToStream(stream);
                stream.Close();
            }
        }

        public void CopyParameterTo(Model other)
        {
            for (var i = 0; i < _parameterNodes.Count; i++)
            {
                other._parameterNodes[i].Output.SetData(_parameterNodes[i].Output.InternalSync());
            }
        }

        public void Forward()
        {
            for (var i = _nodeLayer.Count - 1; i >= 0; i--)
            {
                foreach (var node in _nodeLayer[i]) node.Forward();
            }

            Output.Forward();
        }

        public void Backward()
        {
            Output.Backward();
        }

        public void ClearGradient()
        {
            foreach (var op in _clearGradientOps)
            {
                op.Dispatch();
            }
        }

        private void AddNode(ComputationalNode node, ComputationalNode forwardNode)
        {
            if (_inputNodes.Contains(node)) return;
            var forwardLayer = GetNodeLayerIndex(forwardNode);
            var layer = GetNodeLayerIndex(node);
            var newLayer = Mathf.Max(forwardLayer + 1, layer);
            if (newLayer == layer) return;
            if (layer != -1) _nodeLayer[layer].Remove(node);
            while (_nodeLayer.Count <= newLayer) _nodeLayer.Add(new HashSet<ComputationalNode>());
            _nodeLayer[newLayer].Add(node);
            foreach (var i in node.ReadOnlyInputNodes)
            {
                AddNode(i, node);
            }
        }

        private int GetNodeLayerIndex(ComputationalNode node)
        {
            for (var i = 0; i < _nodeLayer.Count; i++)
            {
                if (_nodeLayer[i].Contains(node)) return i;
            }

            return -1;
        }

        public void Dispose()
        {
            Output.Dispose();
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.Dispose();
                }
            }

            foreach (var op in _clearGradientOps)
            {
                op.Dispose();
            }
        }
    }
}