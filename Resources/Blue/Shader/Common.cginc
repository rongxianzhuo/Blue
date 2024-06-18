

uint calculate_new_index(const uint dim
    , uint old_id
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride)
{
    uint new_id = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        new_id += d * new_stride[i];
    }
    return new_id;
}


uint2 calculate_new_index(const uint dim
    , uint old_id
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride1
    , const StructuredBuffer<uint> new_stride2)
{
    uint new_id1 = 0;
    uint new_id2 = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        new_id1 += d * new_stride1[i];
        new_id2 += d * new_stride2[i];
    }
    return uint2(new_id1, new_id2);
}

uint calculate_new_index(const uint dim
    , uint old_id
    , const uint ignore_dim
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride)
{
    uint new_id = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        if (i != ignore_dim) new_id += d * new_stride[i];
    }
    return new_id;
}

uint calculate_dim_value(const uint dim_size
    , const uint dim
    , uint id
    , const StructuredBuffer<uint> stride)
{
    for (uint i = 0; i < dim_size; i++)
    {
        const uint s = stride[i];
        const uint d = id / s;
        if (i == dim) return d;
        id -= d * s;
    }
    return 0;
}
