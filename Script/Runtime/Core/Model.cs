using System.Collections.Generic;
using System.IO;
using System.Linq;
using Blue.Graph;
using Blue.Kit;
using UnityEngine;

namespace Blue.Core
{
    public class Model
    {

        public readonly GraphNode Output;

        private readonly GraphNode[] _inputNodes;
        private readonly List<TensorNode> _parameterNodes = new List<TensorNode>();
        private readonly List<Operate> _clearGradientOps = new List<Operate>();
        private readonly List<HashSet<GraphNode>> _nodeLayer = new List<HashSet<GraphNode>>();

        public IReadOnlyCollection<TensorNode> ParameterNodes => _parameterNodes;

        public Model(GraphNode outputNode, params GraphNode[] inputNodes)
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
                    if (node is TensorNode dataNode && dataNode.IsParameter) _parameterNodes.Add(dataNode);
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
                    node.GetOutput().LoadFromStream(stream);
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
                node.GetOutput().SaveToStream(stream);
                stream.Close();
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

        public void Destroy()
        {
            Output.Destroy();
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.Destroy();
                }
            }

            foreach (var op in _clearGradientOps)
            {
                op.Destroy();
            }
        }

        public void ClearGradient()
        {
            foreach (var op in _clearGradientOps)
            {
                op.Dispatch();
            }
        }

        private void AddNode(GraphNode node, GraphNode forwardNode)
        {
            if (_inputNodes.Contains(node)) return;
            var forwardLayer = GetNodeLayerIndex(forwardNode);
            var layer = GetNodeLayerIndex(node);
            var newLayer = Mathf.Max(forwardLayer + 1, layer);
            if (newLayer == layer) return;
            if (layer != -1) _nodeLayer[layer].Remove(node);
            while (_nodeLayer.Count <= newLayer) _nodeLayer.Add(new HashSet<GraphNode>());
            _nodeLayer[newLayer].Add(node);
            foreach (var i in node.ReadOnlyInputNodes)
            {
                AddNode(i, node);
            }
        }

        private int GetNodeLayerIndex(GraphNode node)
        {
            for (var i = 0; i < _nodeLayer.Count; i++)
            {
                if (_nodeLayer[i].Contains(node)) return i;
            }

            return -1;
        }
    }
}