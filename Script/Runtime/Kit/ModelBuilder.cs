using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public class ModelBuilder
    {

        private readonly List<IGraphNode> _inputNodes = new List<IGraphNode>(); 
        private readonly Stack<IGraphNode> _inputNodeStack = new Stack<IGraphNode>();

        private IGraphNode _outputNode;

        private int _nextTensorNodeId;

        public ModelBuilder Input(IGraphNode node)
        {
            _outputNode = node;
            _inputNodeStack.Push(node);
            _inputNodes.Add(node);
            return this;
        }

        public ModelBuilder Tensor(bool isParameter, out TensorNode node, params int[] size)
        {
            var id = 0;
            if (isParameter) id = _nextTensorNodeId++;
            node = new TensorNode(id, isParameter, size);
            if (!isParameter) _inputNodes.Add(node);
            Any(node);
            return this;
        }

        public ModelBuilder Random(int inputCount, int outputCount)
        {
            var node = new TensorNode(_nextTensorNodeId++, true, inputCount, outputCount);
            var size = inputCount * outputCount;
            var min = -Mathf.Sqrt(1f / (inputCount + outputCount));
            var max = -min;
            var array = new float[size];
            for (var i = 0; i < size; i++)
            {
                array[i] = UnityEngine.Random.Range(min, max);
            }
            node.GetOutput().SetData(array);
            Any(node);
            return this;
        }

        public ModelBuilder Activation(string activation)
        {
            switch (activation)
            {
                case "relu":
                    Any(OperateNode.ReLU(_inputNodeStack.Pop()));
                    break;
                case "elu":
                    Any(OperateNode.ELU(_inputNodeStack.Pop()));
                    break;
                case "sigmoid":
                    Any(OperateNode.Sigmoid(_inputNodeStack.Pop()));
                    break;
            }
            return this;
        }

        public ModelBuilder Concat()
        {
            var list = new List<IGraphNode>();
            while (_inputNodeStack.Count > 0) list.Insert(0, _inputNodeStack.Pop());
            var concatNode = new ConcatNode(list.ToArray());
            Any(concatNode);
            return this;
        }

        public ModelBuilder Linear(int size)
        {
            Random(_inputNodeStack.Peek().GetOutput().Size[1], size);
            Tensor(true, out _, size);
            var bias = _inputNodeStack.Pop();
            var weight = _inputNodeStack.Pop();
            var input = _inputNodeStack.Pop();
            Any(new LinearNode(input, weight, bias));
            return this;
        }

        public ModelBuilder Dropout(float dropout)
        {
            Any(new DropoutNode(_inputNodeStack.Pop(), dropout));
            return this;
        }

        public Model Build()
        {
            return new Model(_outputNode, _inputNodes.ToArray());
        }

        private void Any(IGraphNode node)
        {
            _outputNode = node;
            _inputNodeStack.Push(node);
            return;
        }

    }
}