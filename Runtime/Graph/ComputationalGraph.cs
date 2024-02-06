using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Blue.Core;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalGraph : IDisposable
    {

        private readonly Dictionary<int, ComputationalNode> _parameterNodes = new Dictionary<int, ComputationalNode>();
        private readonly List<HashSet<ComputationalNode>> _nodeLayer = new List<HashSet<ComputationalNode>>();

        public IEnumerable<ComputationalNode> ParameterNodes => _parameterNodes.Values;

        public ComputationalNode Output => _nodeLayer[_nodeLayer.Count - 1].First();

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
            node.AddBackwardOperate(new Operate("Common/GradientIncrease", "CSMain")
                .SetFloat("weight_decay", 0.000f)
                .SetTensor("gradient", node.Gradient)
                .SetTensor("weight", node)
                .SetTensor("total_gradient", node.TotalGradient)
                .SetDispatchSize(node.TotalGradient.FlattenSize));
            return node;
        }

        public ComputationalNode InputNode(params int[] shape)
        {
            return GeneralNode(false, null, shape);
        }

        public ComputationalNode GeneralNode(bool isParameter, ComputationalNode[] inputNodes, params int[] shape)
        {
            var node = new ComputationalNode(this, isParameter, inputNodes, shape);
            if (isParameter) _parameterNodes.Add(_parameterNodes.Count + 1, node);
            while (_nodeLayer.Count <= node.Layer) _nodeLayer.Add(new HashSet<ComputationalNode>());
            _nodeLayer[node.Layer].Add(node);
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