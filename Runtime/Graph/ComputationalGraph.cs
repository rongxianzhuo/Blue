using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalGraph : IDisposable
    {

        private readonly Dictionary<int, ComputationalNode> _parameterNodes = new Dictionary<int, ComputationalNode>();
        private readonly List<HashSet<ComputationalNode>> _nodeLayer = new List<HashSet<ComputationalNode>>();

        public readonly int BatchSize;

        public IEnumerable<ComputationalNode> ParameterNodes => _parameterNodes.Values;

        public ComputationalNode Output { get; private set; }

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
            while (_nodeLayer.Count <= node.Layer) _nodeLayer.Add(new HashSet<ComputationalNode>());
            _nodeLayer[node.Layer].Add(node);
            if (node.Layer == _nodeLayer.Count - 1) Output = node;
            return node;
        }

        public void Forward()
        {
            foreach (var hashSet in _nodeLayer)
            {
                foreach (var node in hashSet)
                {
                    node.Forward();
                }
            }
        }

        public void Backward()
        {
            Output.Backward();
        }

        public void ClearGradient()
        {
            foreach (var hashSet in _nodeLayer)
            {
                foreach (var node in hashSet)
                {
                    node.ClearGradient();
                }
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
            for (var i = 0; i < _nodeLayer.Count; i++)
            {
                foreach (var node in _nodeLayer[i])
                {
                    node.Dispose();
                }
            }
            _nodeLayer.Clear();
        }
    }
}