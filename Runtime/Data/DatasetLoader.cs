using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Kit;

namespace Blue.Data
{
    public class DatasetLoader : IDisposable
    {

        public readonly int BatchCount;

        private readonly int _batchSize;
        private readonly List<int> _shuffleIndex = new List<int>();
        private readonly List<float[]> _shuffleSample = new List<float[]>();
        private readonly Dictionary<Tensor, Tensor> _tensors = new Dictionary<Tensor, Tensor>();
        private readonly HashSet<KeyValuePair<int, Operate>> _ops = new HashSet<KeyValuePair<int, Operate>>();
        private readonly int _srcStartPropertyId = Operate.PropertyId("src_start");

        public DatasetLoader(int batchSize, int sampleCount)
        {
            _batchSize = batchSize;
            BatchCount = sampleCount / batchSize;
            sampleCount = _batchSize * BatchCount;
            for (var i = 0; i < sampleCount; i++)
            {
                _shuffleIndex.Add(i);
                _shuffleSample.Add(null);
            }
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
            var sampleCount = _batchSize * BatchCount;
            if (samples.Count < sampleCount) throw new Exception("Unknown error");

            for (var i = 0; i < sampleCount; i++)
            {
                _shuffleSample[i] = samples[_shuffleIndex[i]];
            }

            if (_tensors.TryGetValue(target, out var tensor))
            {
                tensor.SetData(_shuffleSample);
            }
            else
            {
                var flatten = new List<float>();
                for (var i = 0; i < BatchCount; i++)
                {
                    for (var j = 0; j < _batchSize; j++)
                    {
                        flatten.AddRange(_shuffleSample[i * _batchSize + j]);
                    }
                }
                tensor = new Tensor(flatten.Count);
                tensor.SetData(flatten.ToArray());
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

        public void Dispose()
        {
            foreach (var t in _tensors.Values)
            {
                t.Dispose();
            }
            _tensors.Clear();

            foreach (var pair in _ops)
            {
                pair.Value.Dispose();
            }
            _ops.Clear();
        }

        public void Shuffle()
        {
            for (var i = _shuffleIndex.Count - 1; i >= 0; i--)
            {
                var j = UnityEngine.Random.Range(0, i + 1);
                (_shuffleIndex[i], _shuffleIndex[j]) = (_shuffleIndex[j], _shuffleIndex[i]);
            }
        }
    }
}