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

        private readonly Dictionary<string, IGraphNode> _allNodes = new Dictionary<string, IGraphNode>();

        private readonly List<TensorNode> _inputNodes = new List<TensorNode>();

        private IGraphNode _outputNode;

        public IGraphNode GetNode(string name) => _allNodes[name];

        public void AddNode(string name, IGraphNode node)
        {
            _allNodes[name] = node;
            _outputNode = node;
        }

        public ModelBuilder TensorNode(string name, int size, bool isParameter)
        {
            var input = new TensorNode(name, size, isParameter);
            if (!isParameter) _inputNodes.Add(input);
            AddNode(name, input);
            return this;
        }

        public ModelBuilder WeightNode(string name, int inputCount, int outputCount)
        {
            var size = inputCount * outputCount;
            var input = new TensorNode(name, size, true);
            var min = -Mathf.Sqrt(1f / (inputCount + outputCount));
            var max = -min;
            var array = new float[size];
            for (var i = 0; i < size; i++)
            {
                array[i] = Random.Range(min, max);
            }
            input.GetOutput().SetData(array);
            AddNode(name, input);
            return this;
        }

        public ModelBuilder MatMulNode(string name, string left, string right)
        {
            var matMul = new MatMulNode(_allNodes[left], _allNodes[right]);
            AddNode(name, matMul);
            return this;
        }

        public ModelBuilder AddNode(string name, string aName, string bName)
        {
            var a = _allNodes[aName];
            var b = _allNodes[bName];
            var add = new OperateNode("Graph/Add", a.GetOutput().Size
                , new KeyValuePair<string, IGraphNode>("a", a)
                , new KeyValuePair<string, IGraphNode>("b", b));
            AddNode(name, add);
            return this;
        }

        public ModelBuilder ReLUNode(string name, string inputName)
        {
            var input = _allNodes[inputName];
            AddNode(name, OperateNode.ReLU(input));
            return this;
        }

        public ModelBuilder EluNode(string name, string inputName)
        {
            var input = _allNodes[inputName];
            AddNode(name, OperateNode.ELU(input));
            return this;
        }

        public ModelBuilder SigmoidNode(string name, string inputName)
        {
            var input = _allNodes[inputName];
            AddNode(name, OperateNode.Sigmoid(input));
            return this;
        }

        public ModelBuilder ConcatLayer(string name, params string[] inputNames)
        {
            var inputs = new IGraphNode[inputNames.Length];
            for (var i = 0; i < inputs.Length; i++)
            {
                inputs[i] = _allNodes[inputNames[i]];
            }

            AddNode(name, new ConcatNode(inputs));
            return this;
        }

        public ModelBuilder DenseLayer(string name, string inputNodeName, int size, string activation=null)
        {
            WeightNode($"{name}.weight", _allNodes[inputNodeName].GetOutput().Size, size);
            MatMulNode($"{name}.matmul", inputNodeName, $"{name}.weight");
            TensorNode($"{name}.bias", size, true);
            if (string.IsNullOrEmpty(activation))
            {
                AddNode(name, $"{name}.matmul", $"{name}.bias");
            }
            else
            {
                AddNode($"{name}.add", $"{name}.matmul", $"{name}.bias");
                switch (activation)
                {
                    case "relu":
                        ReLUNode(name, $"{name}.add");
                        break;
                    case "elu":
                        EluNode(name, $"{name}.add");
                        break;
                    case "sigmoid":
                        SigmoidNode(name, $"{name}.add");
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
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