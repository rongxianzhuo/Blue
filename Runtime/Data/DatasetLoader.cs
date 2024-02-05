using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Data
{
    public class DatasetLoader
    {

        public readonly int BatchCount;

        private readonly int _batchSize;
        private readonly Dictionary<Tensor, Tensor> _tensors = new Dictionary<Tensor, Tensor>();
        private readonly HashSet<KeyValuePair<int, Operate>> _ops = new HashSet<KeyValuePair<int, Operate>>();
        private readonly int _srcStartPropertyId = Operate.PropertyId("src_start");

        public DatasetLoader(int batchSize, int sampleCount)
        {
            _batchSize = batchSize;
            BatchCount = sampleCount / batchSize;
        }

        public void LoadBatch(int batchIndex)
        {
            foreach (var pair in _ops)
            {
                pair.Value.SetInt(_srcStartPropertyId, batchIndex * _batchSize * pair.Key).Dispatch();
            }
        }

        public void LoadDataset(IReadOnlyList<float[]> samples, Tensor target)
        {
            if (samples.Count != _batchSize * BatchCount) throw new Exception("Unknown error");

            if (_tensors.TryGetValue(target, out var tensor))
            {
                tensor.SetData(samples);
            }
            else
            {
                var flatten = new List<float>();
                for (var i = 0; i < BatchCount; i++)
                {
                    for (var j = 0; j < _batchSize; j++)
                    {
                        flatten.AddRange(samples[i * _batchSize + j]);
                    }
                }
                tensor = new Tensor(flatten);
                _tensors[target] = tensor;
                var op = Op.Copy(tensor
                    , 0
                    , 0
                    , target
                    , 0
                    , 0
                    , _batchSize * samples[0].Length
                    , _batchSize * samples[0].Length);
                _ops.Add(new KeyValuePair<int, Operate>(samples[0].Length, op));
            }
        }

        public void Destroy()
        {
            foreach (var t in _tensors.Values)
            {
                t.Release();
            }
            _tensors.Clear();

            foreach (var pair in _ops)
            {
                pair.Value.Destroy();
            }
            _ops.Clear();
        }
        
    }
}