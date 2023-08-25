using System.Collections.Generic;
using System.IO;
using Blue.Operates;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Graph
{

    public class Model
    {

        private const int DefaultBatchSize = 32;

        private readonly List<HashSet<IGraphNode>> _nodeLayer = new List<HashSet<IGraphNode>>();

        private IOptimizer _optimizer;
        private int _batchSize = DefaultBatchSize;
        private int _paramsUpdateFlag = DefaultBatchSize;
        private int _requestBatchSize = DefaultBatchSize;

        public bool IsLoaded => _nodeLayer.Count > 0;

        public int BatchSize
        {
            set
            {
                if (_requestBatchSize == value) return;
                _requestBatchSize = Mathf.Max(1, value);
                if (IsLoaded) return;
                _batchSize = _requestBatchSize;
                _paramsUpdateFlag = _requestBatchSize;
            }
        }

        protected virtual void OnLoad()
        {
            
        }

        protected virtual void OnUnload()
        {
            
        }

        public void Load(IGraphNode outputNode, IOptimizer optimizer)
        {
            Unload();
            _optimizer = optimizer;
            _nodeLayer.Add(new HashSet<IGraphNode>());
            _nodeLayer[0].Add(outputNode);
            outputNode.ForeachInputNode(input => AddNode(input, outputNode));
            OnLoad();
        }

        public void LoadParameterFile(string dirPath)
        {
            foreach (var hashSet in _nodeLayer)
            {
                foreach (var node in hashSet)
                {
                    if (node is DataNode dataNode && dataNode.IsParameter)
                    {
                        dataNode.LoadFromText(File.ReadAllText($"{dirPath}/{dataNode.Name}.bytes"));
                    }
                }
            }
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            foreach (var hashSet in _nodeLayer)
            {
                foreach (var node in hashSet)
                {
                    if (node is DataNode dataNode && dataNode.IsParameter)
                    {
                        var text = dataNode.SaveAsText();
                        File.WriteAllText($"{dirPath}/{dataNode.Name}.bytes", text);
                    }
                }
            }
        }

        public void Unload()
        {
            OnUnload();
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

        public void UpdateParams()
        {
            _paramsUpdateFlag--;
            var shouldUpdateParams = _paramsUpdateFlag <= 0;
            if (shouldUpdateParams) _paramsUpdateFlag = _requestBatchSize;
            else return;
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    var dataNode = node as DataNode;
                    if (dataNode == null || dataNode.TotalGradient == null) continue;
                    TransformOperate.Calculate(dataNode.TotalGradient, 1f / _batchSize, 0);
                    _optimizer.OnBackwardPropagation(dataNode.GetOutput(), dataNode.TotalGradient);
                    SetOperate.Calculate(dataNode.TotalGradient, 0f);
                }
            }

            _batchSize = _requestBatchSize;
        }

        public void BackwardPropagation()
        {
            foreach (var nodes in _nodeLayer)
            {
                foreach (var node in nodes)
                {
                    node.GradientPropagation();
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