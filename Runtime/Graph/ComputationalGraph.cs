using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalGraph : IDisposable
    {

        private readonly Dictionary<int, ComputationalNode> _parameterNodes = new Dictionary<int, ComputationalNode>();
        private readonly List<ComputationalNode> _nodes = new List<ComputationalNode>();

        public readonly int BatchSize;

        public IEnumerable<ComputationalNode> ParameterNodes => _parameterNodes.Values;

        public ComputationalNode Output => _nodes[_nodes.Count - 1];

        public ComputationalGraph(int batchSize)
        {
            BatchSize = batchSize;
        }

        public void LoadParameterFile(string dirPath)
        {
            foreach (var pair in _parameterNodes)
            {
                if (File.Exists(Path.Combine(dirPath, $"{pair.Key}.bytes")))
                {
                    using var stream = File.OpenRead(Path.Combine(dirPath, $"{pair.Key}.bytes"));
                    pair.Value.LoadFromStream(stream);
                    stream.Close();
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {pair.Key}");
                }
            }
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            foreach (var pair in _parameterNodes)
            {
                using var stream = File.OpenWrite(Path.Combine(dirPath, $"{pair.Key}.bytes"));
                pair.Value.SaveToStream(stream);
                stream.Close();
            }
        }

        public ComputationalNode ParameterNode(params int[] shape)
        {
            var node = GeneralNode(true, null, shape);
            return node;
        }

        public ComputationalNode InputNode(int size)
        {
            return GeneralNode(false, null, BatchSize, size);
        }

        public ComputationalNode GeneralNode(bool isParameter, ComputationalNode[] inputNodes, params int[] shape)
        {
            var node = new ComputationalNode(this, inputNodes, shape);
            if (isParameter) _parameterNodes.Add(_parameterNodes.Count + 1, node);
            AddNode(node);
            return node;
        }

        private void AddNode(ComputationalNode node)
        {
            if (node.InputNodes.Count == 0)
            {
                _nodes.Insert(0, node);
                return;
            }

            var i = _nodes.Count - 1;
            while (i >= 0)
            {
                if (node.InputNodes.Contains(_nodes[i])) break;
                i--;
            }
            _nodes.Insert(i + 1, node);
        }

        public void Forward()
        {
            foreach (var node in _nodes)
            {
                node.Forward();
            }
        }

        public void Backward()
        {
            Output.Backward();
        }

        public void ClearGradient()
        {
            foreach (var node in _nodes)
            {
                node.ClearGradient();
            }
        }

        public void CopyParameterTo(ComputationalGraph other)
        {
            foreach (var pair in _parameterNodes)
            {
                other._parameterNodes[pair.Key].SetData(pair.Value.InternalSync());
            }
        }

        public void Dispose()
        {
            foreach (var n in _nodes)
            {
                n.Dispose();
            }
        }
    }
}