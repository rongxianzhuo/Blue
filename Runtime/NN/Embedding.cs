using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.NN
{
    public class Embedding : Module
    {

        public readonly int EmbeddingDim;
        public readonly ComputeBuffer Indices;
        public readonly ComputationalNode Weight;

        public Embedding(ComputeBuffer indices, int embeddingNum, int embeddingDim)
        {
            EmbeddingDim = embeddingDim;
            Indices = indices;
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
            var embedding = new ComputationalNode(new []{Weight}, Indices.count, EmbeddingDim);
            embedding.AddForwardOperate(new Operate("Common/Embedding", "Forward")
                .SetInt("dim", EmbeddingDim)
                .SetBuffer("indices", Indices)
                .SetTensor("weight", Weight)
                .SetTensor("output", embedding)
                .SetDispatchSize(Indices.count));
            
            if (Weight.Gradient != null) embedding.AddBackwardOperate(new Operate("Common/Embedding", "Backward")
                .SetInt("dim", EmbeddingDim)
                .SetInt("len", Indices.count)
                .SetBuffer("indices", Indices)
                .SetTensor("output_gradient", embedding.Gradient)
                .SetTensor("weight_gradient", Weight.Gradient)
                .SetDispatchSize(Weight.Gradient.FlattenSize));
            return embedding;
        }
    }
}