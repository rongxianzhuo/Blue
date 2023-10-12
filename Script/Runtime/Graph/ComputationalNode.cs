using System;
using Blue.Core;
using Blue.Kit;
using UnityEditor;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalNode : GraphNode
    {

        public readonly int Id;
        public readonly ComputationalGraph Graph;
        public readonly Tensor Output;
        public readonly Tensor Gradient;
        public readonly Tensor TotalGradient;

        public bool IsParameter => Id > 0;

        public ComputationalNode(bool isParameter, params int[] shape)
        {
            Graph = new ComputationalGraph();
            if (isParameter)
            {
                Id = Graph.AllocateParameterId();
                TotalGradient = CreateTensor(shape);
            }
            else
            {
                Id = 0;
                TotalGradient = null;
            }
            Output = CreateTensor(shape);
            Gradient = CreateTensor(shape);
        }

        public ComputationalNode(ComputationalGraph graph, bool isParameter, params int[] shape)
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
            Graph = graph;
            Output = CreateTensor(shape);
            Gradient = CreateTensor(shape);
        }
        
        public override Tensor GetOutput()
        {
            return Output;
        }

        public override Tensor GetGradient()
        {
            return Gradient;
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
            var activation = new ComputationalNode(Graph, false, Output.Size);
            
            activation.AddInputNode(this);
            
            activation.AddForwardOperate(new Operate(shaderName, "Forward")
                .SetTensor("rw_output", activation.Output)
                .SetTensor("input", Output)
                .SetDispatchSize(Output.FlattenSize));
            
            activation.AddBackwardOperate(new Operate(shaderName, "Backward_input")
                .SetTensor("r_output", activation.Output)
                .SetTensor("input_gradient", Gradient)
                .SetTensor("output_gradient", activation.Gradient)
                .SetDispatchSize(Gradient.FlattenSize));

            return activation;
        }

        public ComputationalNode Linear(int size, bool newGraph=false)
        {
            var graph = newGraph ? new ComputationalGraph() : Graph;
            var batchSize = Output.Size[0];
            var weight = graph.ParameterNode(Output.Size[1], size);
            var bias = graph.ParameterNode(size);
            var linearNode = new ComputationalNode(graph, false, batchSize, size);
            var tInput = linearNode.CreateTensor(Output.TransposeSize());
            var tWeight = linearNode.CreateTensor(weight.Output.TransposeSize());
            var tBias = linearNode.CreateTensor(1, batchSize);
            Op.Clear(tBias, 1f / batchSize).Dispatch().Destroy();
            linearNode.AddInputNode(this);
            linearNode.AddInputNode(weight);
            linearNode.AddInputNode(bias);
            
            var min = -Mathf.Sqrt(1f / (Output.Size[1] + size));
            var max = -min;
            var array = new float[weight.Output.FlattenSize];
            for (var i = 0; i < weight.Output.FlattenSize; i++)
            {
                array[i] = UnityEngine.Random.Range(min, max);
            }
            weight.Output.SetData(array);
            
            linearNode.AddForwardOperate(Op.MatMul(Output
                , weight.Output
                , linearNode.Output));
            linearNode.AddForwardOperate(Op.Increment(linearNode.Output, bias.Output));
            
            linearNode.AddBackwardOperate(Op.Transpose(weight.Output
                , tWeight));
            linearNode.AddBackwardOperate(Op.MatMul(linearNode.Gradient
                , tWeight
                , Gradient));
            
            linearNode.AddBackwardOperate(Op.Transpose(Output
                , tInput));
            linearNode.AddBackwardOperate(Op.MatMul(tInput
                , linearNode.Gradient
                , weight.Gradient));
            
            linearNode.AddBackwardOperate(Op.Translate(weight.Gradient, 1f / Output.Size[0], 0f));
            linearNode.AddBackwardOperate(Op.MatMul(tBias, linearNode.Gradient, bias.Gradient));

            return linearNode;
        }

        public ComputationalNode Dropout(float dropout)
        {
            var dropoutNode = new ComputationalNode(Graph, false, Output.Size);
            dropoutNode.AddInputNode(this);
            var weightArray = new float[Output.FlattenSize];
            var weight = dropoutNode.CreateTensor(Output.Size);
            dropoutNode.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= dropout ? 1f : 0f;
                }
                weight.SetData(weightArray);
            }));
            dropoutNode.AddForwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", Output)
                .SetTensor("b", weight)
                .SetTensor("result", dropoutNode.Output)
                .SetDispatchSize(Output.FlattenSize));
            dropoutNode.AddBackwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", dropoutNode.Gradient)
                .SetTensor("b", weight)
                .SetTensor("result", Gradient)
                .SetDispatchSize(Gradient.FlattenSize));

            return dropoutNode;
        }
    }
}