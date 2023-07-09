using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Blue.Data
{

    public class MnistData
    {

        public class SingleData
        {
            
            public readonly int SizeX;
            public readonly int SizeY;
            public readonly byte Label;
            public readonly float[] LabelArray = new float[10];
            public readonly float[] ImageData;

            public SingleData(byte label, int sizeX, int sizeY)
            {
                Label = label;
                for (var i = 0; i < 10; i++)
                {
                    LabelArray[i] = label == i ? 1 : 0;
                }
                SizeX = sizeX;
                SizeY = sizeY;
                ImageData = new float[sizeX * sizeY];
            }

            public void Print(StringBuilder builder)
            {
                builder.Append(Label).Append('\n');
                for (var y = 0; y < SizeY; y++)
                {
                    for (var x = 0; x < SizeX; x++)
                    {
                        var i = ImageData[y * SizeX + x] > 0.1f ? 8 : 0;
                        builder.Append(i);
                    }
                    builder.Append('\n');
                }
            }
            
        }

        private bool _isLoaded;

        private readonly List<SingleData> _trainData = new List<SingleData>();

        private readonly List<SingleData> _testData = new List<SingleData>();

        public IReadOnlyList<SingleData> TrainData => _trainData;

        public IReadOnlyList<SingleData> TestData => _testData;

        public void Load(string trainLabelPath, string trainImagePath, string testLabelPath, string testImagePath)
        {
            if (_isLoaded) return;
            _isLoaded = true;
            LoadSample(trainLabelPath, trainImagePath, _trainData);
            LoadSample(testLabelPath, testImagePath, _testData);
        }

        private void LoadSample(string labelPath, string imagePath, List<SingleData> target)
        {
            var trainLabelBytes = File.ReadAllBytes(labelPath);
            var trainImageBytes = File.ReadAllBytes(imagePath);
            var count = trainLabelBytes[7] + trainLabelBytes[6] * 256;
            var labelPosition = 8;
            var imagePosition = 16;
            for (var i = 0; i < count; i++)
            {
                var label = trainLabelBytes[labelPosition];
                var sizeX = trainImageBytes[11] + trainImageBytes[10] * 256;
                var sizeY = trainImageBytes[15] + trainImageBytes[14] * 256;
                var singleData = new SingleData(label, sizeX, sizeY);
                var imageSize = sizeX * sizeY;
                for (var j = 0; j < imageSize; j++)
                {
                    singleData.ImageData[j] = trainImageBytes[imagePosition++] / 255f;
                }
                labelPosition++;
                target.Add(singleData);
            }
        }

    }

}