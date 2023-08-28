using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Blue.Graph
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
                node.LoadFromText(File.ReadAllText($"{dirPath}/{node.Name}.bytes"));
            });
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            ForeachParameterNode(node =>
            {
                var text = node.SaveAsText();
                File.WriteAllText($"{dirPath}/{node.Name}.bytes", text);
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