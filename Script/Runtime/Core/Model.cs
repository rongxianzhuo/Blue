using System.Collections.Generic;
using System.IO;
using Blue.Graph;
using Blue.Optimizers;
using UnityEngine;

namespace Blue.Core
{

    public class Model
    {
        
        private static Operate _translateOperate;

        private static Operate GetTranslateOperate() => _translateOperate ??= new Operate("Common/Translate", "CSMain"
            , "weight", "bias", "rw_buffer1");

        private const int DefaultBatchSize = 32;

        private readonly List<HashSet<IGraphNode>> _nodeLayer = new List<HashSet<IGraphNode>>();

        private IOptimizer _optimizer;
        private int _batchSize = DefaultBatchSize;
        private int _paramsUpdateFlag = DefaultBatchSize;
        private int _requestBatchSize = DefaultBatchSize;
        private Operate _lossFunction;

        public IGraphNode Output { get; private set; }

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

        public void Load(IGraphNode outputNode, IOptimizer optimizer, string lossFunction)
        {
            Unload();
            Output = outputNode;
            _optimizer = optimizer;
            _nodeLayer.Add(new HashSet<IGraphNode>());
            _nodeLayer[0].Add(outputNode);
            outputNode.ForeachInputNode(input => AddNode(input, outputNode));
            _lossFunction = new Operate($"LossFunction/{lossFunction}", "CSMain"
                , "output", "target", "gradient");
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
            Output = null;
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
                    
                    GetTranslateOperate().CreateTask()
                        .SetFloat(1f / _batchSize)
                        .SetFloat(0)
                        .SetBuffer(dataNode.TotalGradient)
                        .Dispatch(new Vector3Int(dataNode.GetOutput().count, 1, 1));
                    _optimizer.OnBackwardPropagation(dataNode.GetOutput(), dataNode.TotalGradient);
                    GetTranslateOperate().CreateTask()
                        .SetFloat(0)
                        .SetFloat(0)
                        .SetBuffer(dataNode.TotalGradient)
                        .Dispatch(new Vector3Int(dataNode.TotalGradient.count, 1, 1));
                }
            }

            _batchSize = _requestBatchSize;
        }

        public void BackwardPropagation(ComputeBuffer target)
        {
            _lossFunction.CreateTask()
                .SetBuffer(Output.GetOutput())
                .SetBuffer(target)
                .SetBuffer(Output.GetGradient())
                .Dispatch(new Vector3Int(target.count, 1, 1));
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