using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Blue.Kit
{
    public class ModelBuilder
    {

        private readonly List<TensorNode> _inputNodes = new List<TensorNode>();

        private readonly Stack<IGraphNode> _inputNodeStack = new Stack<IGraphNode>();

        private IGraphNode _outputNode;

        private int _nextTensorNodeId;

        private void AddNode(IGraphNode node)
        {
            _outputNode = node;
            _inputNodeStack.Push(node);
        }

        public ModelBuilder TensorNode(int size, bool isParameter, out TensorNode node)
        {
            node = new TensorNode(_nextTensorNodeId++, size, isParameter);
            if (!isParameter) _inputNodes.Add(node);
            AddNode(node);
            return this;
        }

        public ModelBuilder WeightNode(int inputCount, int outputCount, out TensorNode node)
        {
            var size = inputCount * outputCount;
            node = new TensorNode(_nextTensorNodeId++, size, true);
            var min = -Mathf.Sqrt(1f / (inputCount + outputCount));
            var max = -min;
            var array = new float[size];
            for (var i = 0; i < size; i++)
            {
                array[i] = Random.Range(min, max);
            }
            node.GetOutput().SetData(array);
            AddNode(node);
            return this;
        }

        public ModelBuilder MatMulNode()
        {
            var right = _inputNodeStack.Pop();
            var left = _inputNodeStack.Pop();
            var matMul = new MatMulNode(left, right);
            AddNode(matMul);
            return this;
        }

        public ModelBuilder AddNode()
        {
            var a = _inputNodeStack.Pop();
            var b = _inputNodeStack.Pop();
            var add = new OperateNode("Graph/Add", a.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("a", a)
                , new KeyValuePair<string, IGraphNode>("b", b));
            AddNode(add);
            return this;
        }

        public ModelBuilder ReLUNode()
        {
            AddNode(OperateNode.ReLU(_inputNodeStack.Pop()));
            return this;
        }

        public ModelBuilder EluNode()
        {
            AddNode(OperateNode.ELU(_inputNodeStack.Pop()));
            return this;
        }

        public ModelBuilder SigmoidNode()
        {
            AddNode(OperateNode.Sigmoid(_inputNodeStack.Pop()));
            return this;
        }

        public ModelBuilder ConcatLayer()
        {
            var inputs = new IGraphNode[_inputNodeStack.Count];
            for (var i = inputs.Length - 1; i >= 0; i--)
            {
                inputs[i] = _inputNodeStack.Pop();
            }

            AddNode(new ConcatNode(inputs));
            return this;
        }

        public ModelBuilder DenseLayer(int size, string activation=null)
        {
            WeightNode(_inputNodeStack.Peek().GetOutput().Size, size, out _);
            MatMulNode();
            TensorNode(size, true, out _);
            AddNode();
            if (string.IsNullOrEmpty(activation)) return this;
            switch (activation)
            {
                case "relu":
                    ReLUNode();
                    break;
                case "elu":
                    EluNode();
                    break;
                case "sigmoid":
                    SigmoidNode();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
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