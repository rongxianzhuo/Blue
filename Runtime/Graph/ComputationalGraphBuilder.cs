using System;
using Blue.Core;
using Blue.Kit;
using UnityEngine;

namespace Blue.Graph
{
    public static class ComputationalGraphBuilder
    {

        public static ComputationalNode Activation(this ComputationalNode node, string activationName)
        {
            var shaderName = activationName switch
            {
                "relu" => "Graph/ReLU",
                "elu" => "Graph/ELU",
                "sigmoid" => "Graph/Sigmoid",
                _ => throw new Exception("Unknown activation name")
            };
            var activation = node.Graph.GeneralNode(false, new[] { node }, node.Size);
            
            activation.AddForwardOperate(new Operate(shaderName, "Forward")
                .SetTensor("rw_output", activation)
                .SetTensor("input", node)
                .SetDispatchSize(node.FlattenSize));
            
            activation.AddBackwardOperate(new Operate(shaderName, "Backward_input")
                .SetTensor("r_output", activation)
                .SetTensor("input_gradient", node.Gradient)
                .SetTensor("output_gradient", activation.Gradient)
                .SetDispatchSize(node.Gradient.FlattenSize));

            return activation;
        }

        public static ComputationalNode Concat(this ComputationalGraph graph, params ComputationalNode[] nodes)
        {
            var size = 0;
            foreach (var node in nodes)
            {
                size += node.Size[1];
            }
            var concat = graph.GeneralNode(false, nodes, nodes[0].Size[0], size);
            
            var start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t;
                concat.AddForwardOperate(Op.Copy(inputNode, 0, 0
                    , concat, start, size - inputNode.Size[1]
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }
            
            start = 0;
            foreach (var t in nodes)
            {
                var inputNode = t.Gradient;
                concat.AddBackwardOperate(Op.Copy(concat.Gradient, start, size - inputNode.Size[1]
                    , inputNode, 0, 0
                    , inputNode.Size[1]
                    , inputNode.FlattenSize));
                start += inputNode.Size[1];
            }

            return concat;
        }

        public static ComputationalNode Linear(this ComputationalNode node, int size)
        {
            var batchSize = node.Size[0];
            var weight = node.Graph.ParameterNode(node.Size[1], size);
            var bias = node.Graph.ParameterNode(size);
            var linearNode = node.Graph.GeneralNode(false, new []{node, weight, bias}, batchSize, size);
            var tInput = linearNode.CreateTempTensor(node.TransposeSize());
            var tWeight = linearNode.CreateTempTensor(weight.TransposeSize());
            var tBias = linearNode.CreateTempTensor(1, batchSize);
            Op.Clear(tBias, 1f / batchSize).Dispatch().Dispose();
            
            var min = -Mathf.Sqrt(1f / (node.Size[1] + size));
            var max = -min;
            var array = new float[weight.FlattenSize];
            for (var i = 0; i < weight.FlattenSize; i++)
            {
                array[i] = UnityEngine.Random.Range(min, max);
            }
            weight.SetData(array);
            
            linearNode.AddForwardOperate(Op.MatMul(node
                , weight
                , linearNode));
            linearNode.AddForwardOperate(Op.Increment(linearNode, bias));
            
            linearNode.AddBackwardOperate(Op.Transpose(weight
                , tWeight));
            linearNode.AddBackwardOperate(Op.MatMul(linearNode.Gradient
                , tWeight
                , node.Gradient));
            
            linearNode.AddBackwardOperate(Op.Transpose(node
                , tInput));
            linearNode.AddBackwardOperate(Op.Translate(tInput, 1f / node.Size[0], 0f));
            linearNode.AddBackwardOperate(Op.IncreaseMatMul(tInput
                , linearNode.Gradient
                , weight.Gradient));
            
            linearNode.AddBackwardOperate(Op.IncreaseMatMul(tBias, linearNode.Gradient, bias.Gradient));

            return linearNode;
        }

        public static ComputationalNode Dropout(this ComputationalNode node, float dropout)
        {
            var dropoutNode = node.Graph.GeneralNode(false, new []{node}, node.Size);
            var weightArray = new float[node.FlattenSize];
            var weight = dropoutNode.CreateTempTensor(node.Size);
            dropoutNode.AddForwardOperate(new Operate(() =>
            {
                for (var i = 0; i < weightArray.Length; i++)
                {
                    weightArray[i] = UnityEngine.Random.Range(0f, 1f) >= dropout ? 1f : 0f;
                }
                weight.SetData(weightArray);
            }));
            dropoutNode.AddForwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", node)
                .SetTensor("b", weight)
                .SetTensor("result", dropoutNode)
                .SetDispatchSize(node.FlattenSize));
            dropoutNode.AddBackwardOperate(new Operate("Common/Mul", "CSMain")
                .SetTensor("a", dropoutNode.Gradient)
                .SetTensor("b", weight)
                .SetTensor("result", node.Gradient)
                .SetDispatchSize(node.Gradient.FlattenSize));

            return dropoutNode;
        }
        
    }
}