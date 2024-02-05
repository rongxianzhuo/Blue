using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalNode : Tensor
    {

        public readonly int Id;
        public readonly Tensor Gradient;
        public readonly Tensor TotalGradient;
        
        private readonly ComputationalGraph _graph;
        private readonly List<ComputationalNode> _inputNodes = new List<ComputationalNode>();
        private readonly List<Operate> _forwardOperates = new List<Operate>();
        private readonly List<Operate> _backwardOperates = new List<Operate>();
        private readonly HashSet<Tensor> _bindTensors = new HashSet<Tensor>();

        public bool IsParameter => Id > 0;
        
        public IReadOnlyList<ComputationalNode> ReadOnlyInputNodes => _inputNodes;

        public ComputationalNode(bool isParameter, params int[] shape) : base(shape)
        {
            _graph = new ComputationalGraph();
            if (isParameter)
            {
                Id = _graph.AllocateParameterId();
                TotalGradient = CreateTensor(shape);
            }
            else
            {
                Id = 0;
                TotalGradient = null;
            }
            Gradient = CreateTensor(shape);
        }

        public ComputationalNode(ComputationalGraph graph, bool isParameter, params int[] shape) : base(shape)
        {
            if (isParameter)
            {
                Id = graph.AllocateParameterId();
                TotalGradient = CreateTensor(shape);
            }
            else
            {
                Id = 0;
                TotalGradient = null;
            }
            _graph = graph;
            Gradient = CreateTensor(shape);
        }

        private Tensor CreateTensor(params int[] shape)
        {
            var tensor = new Tensor(shape);
            _bindTensors.Add(tensor);
            return tensor;
        }

        public void AddInputNode(params ComputationalNode[] node)
        {
            _inputNodes.AddRange(node);
        }

        public void AddForwardOperate(Operate operate)
        {
            _forwardOperates.Add(operate);
        }

        public void AddBackwardOperate(Operate operate)
        {
            _backwardOperates.Add(operate);
        }

        public void Backward()
        {
            foreach (var o in _backwardOperates)
            {
                o.Dispatch();
            }

            foreach (var node in _inputNodes)
            {
                node.Backward();
            }
        }

        public void Forward()
        {
            foreach (var o in _forwardOperates)
            {
                o.Dispatch();
            }
        }

        public ComputationalNode Activation(string activationName)
        {
            var shaderName = activationName switch
            {
                "relu" => "Graph/ReLU",
                "elu" => "Graph/ELU",
                "sigmoid" => "Graph/Sigmoid",
                _ => throw new Exception("Unknown activation name")
            };
            var activation = new ComputationalNode(_graph, false, Size);
            
            activation.AddInputNode(this);
            
            activation.AddForwardOperate(new Operate(shaderName, "Forward")
                .SetTensor("rw_output", activation)
                .SetTensor("input", this)
                .SetDispatchSize(FlattenSize));
            
            activation.AddBackwardOperate(new Operate(shaderName, "Backward_input")
                .SetTensor("r_output", activation)
                .SetTensor("input_gradient", Gradient)
                .SetTensor("output_gradient", activation.Gradient)
                .SetDispatchSize(Gradient.FlattenSize));

            return activation;
        }

        public ComputationalNode Linear(int size, bool newGraph=false)
        {
            var graph = newGraph ? new ComputationalGraph() : _graph;
            var batchSize = Size[0];
            var weight = graph.ParameterNode(Size[1], size);
            var bias = graph.ParameterNode(size);
            var linearNode = new ComputationalNode(graph, false, batchSize, size);
            var tInput = linearNode.CreateTensor(TransposeSize());
            var tWeight = linearNode.CreateTensor(weight.TransposeSize());
            var tBias = linearNode.CreateTensor(1, batchSize);
            Op.Clear(tBias, 1f / batchSize).Dispatch().Dispose();
            linearNode.AddInputNode(this);
            linearNode.AddInputNode(weight);
            linearNode.AddInputNode(bias);
            
            var min = -Mathf.Sqrt(1f / (Size[1] + size));
            var max = -min;
            var array = new float[weight.FlattenSize];
            for (var i = 0; i < weight.FlattenSize; i++)
            {
                array[i] = UnityEngine.Random.Range(min, max);
            }
            weight.SetData(array);
            
            linearNode.AddForwardOperate(Op.MatMul(this
                , weight
                , linearNode));
            linearNode.AddForwardOperate(Op.Increment(linearNode, bias));
            
            linearNode.AddBackwardOperate(Op.Transpose(weight
                , tWeight));
            linearNode.AddBackwardOperate(Op.MatMul(linearNode.Gradient
                , tWeight
                , Gradient));
            
            linearNode.AddBackwardOperate(Op.Transpose(this
                , tInput));
            linearNode.AddBackwardOperate(Op.MatMul(tInput
                , linearNode.Gradient
                , weight.Gradient));
            
            linearNode.AddBackwardOperate(Op.Translate(weight.Gradient, 1f / Size[0], 0f));
            linearNode.AddBackwardOperate(Op.MatMul(tBias, linearNode.Gradient, bias.Gradient));

            return linearNode;
        }

        public ComputationalNode Dropout(float dropout)
        {
            var dropoutNode = new ComputationalNode(_graph, false, Size);
            dropoutNode.AddInputNode(this);
            var weightArray = new float[FlattenSize];
            var weight = dropoutNode.CreateTensor(Size);
            dropoutNode.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= dropout ? 1f : 0f;
                }
                weight.SetData(weightArray);
            }));
            dropoutNode.AddForwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", this)
                .SetTensor("b", weight)
                .SetTensor("result", dropoutNode)
                .SetDispatchSize(FlattenSize));
            dropoutNode.AddBackwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", dropoutNode.Gradient)
                .SetTensor("b", weight)
                .SetTensor("result", Gradient)
                .SetDispatchSize(Gradient.FlattenSize));

            return dropoutNode;
        }

        public override void Dispose()
        {
            foreach (var o in _forwardOperates)
            {
                o.Dispose();
            }
            _forwardOperates.Clear();
            foreach (var o in _backwardOperates)
            {
                o.Dispose();
            }
            _backwardOperates.Clear();

            foreach (var t in _bindTensors)
            {
                t.Dispose();
            }
            _bindTensors.Clear();
            base.Dispose();
        }
    }
}