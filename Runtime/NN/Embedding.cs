using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.NN
{
    public class Embedding : Module
    {

        public readonly int EmbeddingDim;
        public readonly ComputationalNode Weight;

        public Embedding(int embeddingNum, int embeddingDim)
        {
            EmbeddingDim = embeddingDim;
            Weight = CreateParameter(embeddingNum, embeddingDim);
            
            var weightArray = new float[embeddingNum * embeddingDim];
            for (var i = 0; i < weightArray.Length; i++)
            {
                weightArray[i] = Common.RandN(0, 1);
            }
            Weight.SetData(weightArray);
        }
        
        public override ComputationalNode Build(params ComputationalNode[] input)
        {
            var indices = input[0];
            var embedding = new ComputationalNode(new []{Weight}, indices.FlattenSize, EmbeddingDim);
            embedding.AddForwardOperate(new Operate("Common/Embedding", "Forward")
                .SetInt("dim", EmbeddingDim)
                .SetTensor("indices", indices)
                .SetTensor("weight", Weight)
                .SetTensor("output", embedding)
                .SetDispatchSize(indices.FlattenSize));
            
            if (Weight.Gradient != null) embedding.AddBackwardOperate(new Operate("Common/Embedding", "Backward")
                .SetInt("dim", EmbeddingDim)
                .SetInt("len", indices.FlattenSize)
                .SetTensor("indices", indices)
                .SetTensor("output_gradient", embedding.Gradient)
                .SetTensor("weight_gradient", Weight.Gradient)
                .SetDispatchSize(Weight.Gradient.FlattenSize));
            return embedding;
        }
    }
}