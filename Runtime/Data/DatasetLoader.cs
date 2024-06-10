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
        private readonly int[] _shuffleIndex;
        private readonly Dictionary<Tensor, List<float[]>> _batchData = new Dictionary<Tensor, List<float[]>>();

        public DatasetLoader(int batchSize, int sampleCount)
        {
            _batchSize = batchSize;
            BatchCount = sampleCount / batchSize;
            _shuffleIndex = new int[sampleCount];
            sampleCount = _batchSize * BatchCount;
            for (var i = 0; i < sampleCount; i++)
            {
                _shuffleIndex[i] = i;
            }
        }

        public void LoadBatch(int batchIndex)
        {
            foreach (var pair in _batchData)
            {
                pair.Key.SetData(pair.Value[batchIndex]);
            }
        }

        public void LoadDataset(IReadOnlyList<float[]> samples, Tensor tensor)
        {
            var sampleCount = _batchSize * BatchCount;
            if (samples.Count < sampleCount) throw new Exception("Unknown error");

            var batchData = new List<float[]>();
            for (var i = 0; i < BatchCount; i++)
            {
                var data = new List<float>();
                for (var j = 0; j < _batchSize; j++)
                {
                    data.AddRange(samples[_shuffleIndex[i * _batchSize + j]]);
                }
                batchData.Add(data.ToArray());
            }
            _batchData[tensor] = batchData;
        }

        public void Shuffle()
        {
            for (var i = _shuffleIndex.Length - 1; i > 0; i--)
            {
                var j = UnityEngine.Random.Range(0, i);
                (_shuffleIndex[i], _shuffleIndex[j]) = (_shuffleIndex[j], _shuffleIndex[i]);
            }
        }
    }
}