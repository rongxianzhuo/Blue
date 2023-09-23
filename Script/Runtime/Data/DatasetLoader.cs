using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Data
{
    public class DatasetLoader
    {

        private readonly int _batchSize;
        private readonly int _batchCount;
        private readonly HashSet<Tensor> _tensors = new HashSet<Tensor>();
        private readonly HashSet<KeyValuePair<int, OperateInstance>> _ops = new HashSet<KeyValuePair<int, OperateInstance>>();
        private readonly int _srcStartPropertyId = OperateInstance.PropertyId("src_start");

        public DatasetLoader(int batchSize, int sampleCount)
        {
            _batchSize = batchSize;
            _batchCount = sampleCount / batchSize;
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
            if (samples.Count < _batchSize * _batchCount) throw new Exception("Unknown error");
            var flatten = new List<float>();
            for (var i = 0; i < _batchCount; i++)
            {
                for (var j = 0; j < _batchSize; j++)
                {
                    flatten.AddRange(samples[i * _batchSize + j]);
                }
            }
            var tensor = new Tensor(flatten);
            _tensors.Add(tensor);
            var op = Op.Copy(tensor
                , 0
                , 0
                , target
                , 0
                , 0
                , _batchSize * samples[0].Length
                , _batchSize * samples[0].Length);
            _ops.Add(new KeyValuePair<int, OperateInstance>(samples[0].Length, op));
        }

        public void Destroy()
        {
            foreach (var t in _tensors)
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