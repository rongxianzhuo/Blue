using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.Kit
{
    public class ModelBuilder
    {

        private readonly List<TensorNode> _inputNodes = new List<TensorNode>();

        private readonly Stack<IGraphNode> _inputNodeStack = new Stack<IGraphNode>();

        private IGraphNode _outputNode;

        private int _nextTensorNodeId;

        public void Any(IGraphNode node)
        {
            _outputNode = node;
            _inputNodeStack.Push(node);
        }

        public ModelBuilder Tensor(int size, bool isParameter, out TensorNode node)
        {
            node = new TensorNode(_nextTensorNodeId++, size, isParameter);
            if (!isParameter) _inputNodes.Add(node);
            Any(node);
            return this;
        }

        public ModelBuilder Random(int inputCount, int outputCount)
        {
            var size = inputCount * outputCount;
            var min = -Mathf.Sqrt(1f / (inputCount + outputCount));
            var max = -min;
            var list = new List<float>();
            for (var i = 0; i < size; i++)
            {
                list.Add(UnityEngine.Random.Range(min, max));
            }
            var node = new TensorNode(_nextTensorNodeId++, true, list);
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

        public ModelBuilder Linear(int size)
        {
            Random(_inputNodeStack.Peek().GetOutput().FlattenSize, size);
            Tensor(size, true, out _);
            var bias = _inputNodeStack.Pop();
            var weight = _inputNodeStack.Pop();
            var input = _inputNodeStack.Pop();
            Any(new LinearNode(input, weight, bias, size));
            return this;
        }

        public ModelBuilder ConcatLayer()
        {
            var inputs = new IGraphNode[_inputNodeStack.Count];
            for (var i = inputs.Length - 1; i >= 0; i--)
            {
                inputs[i] = _inputNodeStack.Pop();
            }

            Any(new ConcatNode(inputs));
            return this;
        }

        public Model Build()
        {
            return new Model(_outputNode);
        }

        public SimpleModel BuildSimpleModel()
        {
            return new SimpleModel(_outputNode, _inputNodes.ToArray());
        }

    }
}