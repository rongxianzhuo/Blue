using System;
using System.Collections.Generic;
using System.IO;
using Blue.Graph;
using UnityEngine;

namespace Blue.Core
{
    public class NodeGraph
    {

        public readonly IGraphNode Output;
        
        private readonly List<HashSet<IGraphNode>> _nodeLayer = new List<HashSet<IGraphNode>>();

        public NodeGraph(IGraphNode outputNode)
        {
            Output = outputNode;
            outputNode.ForeachInputNode(input => AddNode(input, outputNode));
        }

        public void LoadParameterFile(string dirPath)
        {
            ForeachParameterNode(node =>
            {
                if (File.Exists($"{dirPath}/{node.Id}.bytes"))
                {
                    using var stream = File.OpenRead($"{dirPath}/{node.Id}.bytes");
                    node.GetOutput().LoadFromStream(stream);
                    stream.Close();
                }
                else if (File.Exists($"{dirPath}/{node.Id}.json"))
                {
                    node.GetOutput().LoadFromJson(File.ReadAllText($"{dirPath}/{node.Id}.json"));
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {node.Id}");
                }
            });
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            ForeachParameterNode(node =>
            {
                using var stream = File.OpenWrite($"{dirPath}/{node.Id}.bytes");
                node.GetOutput().SaveToStream(stream);
                stream.Close();
            });
        }

        public void ForeachParameterNode(Action<TensorNode> action)
        {
            for (var i = _nodeLayer.Count - 1; i >= 0; i--)
            {
                foreach (var node in _nodeLayer[i])
                {
                    if (node is TensorNode dataNode && dataNode.IsParameter) action(dataNode);
                }
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
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes) node.Backward();
            }
        }

        public virtual void Destroy()
        {
            Output.Destroy();
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.Destroy();
                }
            }
        }

        private void AddNode(IGraphNode node, IGraphNode forwardNode)
        {
            var forwardLayer = GetNodeLayerIndex(forwardNode);
            var layer = GetNodeLayerIndex(node);
            var newLayer = Mathf.Max(forwardLayer + 1, layer);
            if (newLayer == layer) return;
            if (layer != -1) _nodeLayer[layer].Remove(node);
            while (_nodeLayer.Count <= newLayer) _nodeLayer.Add(new HashSet<IGraphNode>());
            _nodeLayer[newLayer].Add(node);
            node.ForeachInputNode(input => AddNode(input, node));
        }

        private int GetNodeLayerIndex(IGraphNode node)
        {
            for (var i = 0; i < _nodeLayer.Count; i++)
            {
                if (_nodeLayer[i].Contains(node)) return i;
            }

            return -1;
        }
    }
}