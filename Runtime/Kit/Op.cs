using Blue.Core;
using UnityEngine;

namespace Blue.Kit
{
    public static class Op
    {

        public static Operate Texture2Tensor(Texture texture, Tensor tensor)
        {
            return new Operate("Common/T3", "Texture2Tensor")
                .SetInt("w", texture.width)
                .SetTexture("r_texture", texture)
                .SetTensor("w_tensor", tensor)
                .SetDispatchSize(tensor.FlattenSize / 4);
        }

        public static Operate Tensor2Texture(Tensor tensor, Texture texture)
        {
            return new Operate("Common/T3", "Tensor2Texture")
                .SetInt("w", texture.width)
                .SetTexture("w_texture", texture)
                .SetTensor("r_tensor", tensor)
                .SetDispatchSize(texture.width, texture.height);
        }

        public static Operate Lerp(Tensor a, Tensor b, Tensor t)
        {
            return new Operate("Common/Lerp", "CSMain")
                .SetInt("t_len", t.FlattenSize)
                .SetTensor("t", t)
                .SetTensor("a", a)
                .SetTensor("b", b)
                .SetDispatchSize(a.FlattenSize);
        }
        
        public static Operate Copy(Tensor src, int srcStart, int srcInterval, Tensor dst, int dstStart, int dstInterval, int stride, int length)
        {
            return new Operate("Common/Copy", "CSMain")
                .SetInt("src_start", srcStart)
                .SetInt("dst_start", dstStart)
                .SetInt("src_interval", srcInterval)
                .SetInt("dst_interval", dstInterval)
                .SetInt("stride", stride)
                .SetTensor("src_buffer", src)
                .SetTensor("dst_buffer", dst)
                .SetDispatchSize(length);
        }
        
        public static Operate Clear(Tensor buffer, float clearValue)
        {
            return new Operate("Common/Clear", "CSMain")
                .SetFloat("clear_value", clearValue)
                .SetTensor("buffer", buffer)
                .SetDispatchSize(buffer.FlattenSize);
        }
        
        public static Operate Transpose(Tensor src, Tensor dst)
        {
            return new Operate("Common/Transpose", "CSMain")
                .SetInt("src_height", src.Size[0])
                .SetInt("src_width", src.Size[1])
                .SetTensor("from", src)
                .SetTensor("to", dst)
                .SetDispatchSize(dst.FlattenSize);
        }
        
        public static Operate CrossEntropyLoss(Tensor output, Tensor target, Tensor gradient)
        {
            return new Operate("LossFunction/CrossEntropyLoss", "CSMain")
                .SetInt("total_count", target.Size[1])
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetDispatchSize(target.FlattenSize);
        }
        
        public static Operate WeightedL1Loss(Tensor output, Tensor target, Tensor gradient, Tensor weight)
        {
            return new Operate("LossFunction/WeightedL1Loss", "CSMain")
                .SetTensor("output", output)
                .SetTensor("target", target)
                .SetTensor("gradient", gradient)
                .SetTensor("weight", weight)
                .SetDispatchSize(target.FlattenSize);
        }
        
        public static Operate Variance(Tensor input, Tensor result)
        {
            return new Operate("Common/Variance", "CSMain")
                .SetInt("n", input.Size[1])
                .SetTensor("buffer", input)
                .SetTensor("result", result)
                .SetDispatchSize(input.Size[0]);
        }
        
    }
}