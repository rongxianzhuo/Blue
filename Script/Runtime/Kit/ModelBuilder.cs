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
            var node = new TensorNode(_nextTensorNodeId++, size, true);
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

        public ModelBuilder MatMul()
        {
            var right = _inputNodeStack.Pop();
            var left = _inputNodeStack.Pop();
            var matMul = new MatMulNode(left, right);
            Any(matMul);
            return this;
        }

        public ModelBuilder Add()
        {
            var a = _inputNodeStack.Pop();
            var b = _inputNodeStack.Pop();
            var add = new OperateNode("Graph/Add", a.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("a", a)
                , new KeyValuePair<string, IGraphNode>("b", b));
            Any(add);
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

        public ModelBuilder DenseLayer(int size, string activation = null) =>
            Random(_inputNodeStack.Peek().GetOutput().Size, size)
                .MatMul()
                .Tensor(size, true, out _)
                .Add()
                .Activation(activation);

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