using System.Collections.Generic;
using Blue.Graph;
using Blue.Operates;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Kit
{

    public class Model
    {

        private readonly List<HashSet<IGraphNode>> _nodeLayer = new List<HashSet<IGraphNode>>();

        private IOptimizer _optimizer;
        private int _batchSize;
        private int _paramsUpdateFlag;

        public void Load(IGraphNode outputNode, IOptimizer optimizer, int batchSize)
        {
            Unload();
            _batchSize = batchSize;
            _paramsUpdateFlag = batchSize;
            _optimizer = optimizer;
            _nodeLayer.Add(new HashSet<IGraphNode>());
            _nodeLayer[0].Add(outputNode);
            outputNode.ForeachInputNode(input => AddNode(input, outputNode));
        }

        public void Unload()
        {
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.Destroy();
                }
            }
            _nodeLayer.Clear();
            _optimizer?.Destroy();
            _optimizer = null;
        }

        public void ForwardPropagation()
        {
            for (var i = _nodeLayer.Count - 1; i >= 0; i--)
            {
                foreach (var node in _nodeLayer[i])
                {
                    node.Calculate();
                }
            }
        }

        public void BackwardPropagation()
        {
            _paramsUpdateFlag--;
            var shouldUpdateParams = _paramsUpdateFlag <= 0;
            if (shouldUpdateParams) _paramsUpdateFlag = _batchSize;
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.GradientPropagation();
                    if (!shouldUpdateParams) continue;
                    var dataNode = node as DataNode;
                    if (dataNode == null || dataNode.TotalGradient == null) continue;
                    TransformOperate.Calculate(dataNode.TotalGradient, 1f / _batchSize, 0);
                    _optimizer.OnBackwardPropagation(dataNode.GetOutput(), dataNode.TotalGradient);
                    SetOperate.Calculate(dataNode.TotalGradient, 0f);
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