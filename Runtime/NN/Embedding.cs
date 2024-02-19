using Blue.Core;
using Blue.Graph;
using UnityEngine;

namespace Blue.Runtime.NN
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
            Weight = new ComputationalNode(true, embeddingNum, embeddingDim);
            RegisterParameter(Weight);
            
            var weightArray = new float[embeddingNum * embeddingDim];
            for (var i = 0; i < weightArray.Length; i++)
            {
                weightArray[i] = RandN();
            }
            Weight.SetData(weightArray);
        }
        
        public override ComputationalNode Forward(params ComputationalNode[] input)
        {
            var embedding = new ComputationalNode(new []{Weight}, Indices.count, EmbeddingDim);
            embedding.AddForwardOperate(new Operate("Common/Embedding", "Forward")
                .SetInt("dim", EmbeddingDim)
                .SetBuffer("indices", Indices)
                .SetTensor("weight", Weight)
                .SetTensor("output", embedding)
                .SetDispatchSize(Indices.count));
            
            embedding.AddBackwardOperate(new Operate("Common/Embedding", "Backward")
                .SetInt("dim", EmbeddingDim)
                .SetInt("len", Indices.count)
                .SetBuffer("indices", Indices)
                .SetTensor("output_gradient", embedding.Gradient)
                .SetTensor("weight_gradient", Weight.Gradient)
                .SetDispatchSize(Weight.Gradient.FlattenSize));
            return embedding;
        }

        private static float RandN()
        {
            const float min = -4f;
            const float max = 4f;
            while (true)
            {
                var i = Random.Range(min, max);
                var p = 1f / Mathf.Sqrt(2 * Mathf.PI) * Mathf.Exp(-0.5f * i * i);
                if (Random.Range(0f, 1f) > p) continue;
                return i;
            }
        }
    }
}