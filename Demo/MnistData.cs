using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using System.IO.Compression;

namespace Blue.Demo
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

        private readonly List<float[]> _trainInputData = new List<float[]>();

        private readonly List<float[]> _trainOutputData = new List<float[]>();

        private readonly List<float[]> _testInputData = new List<float[]>();

        private readonly List<float[]> _testOutputData = new List<float[]>();

        private readonly List<int> _testOutputLabel = new List<int>();

        public IReadOnlyList<float[]> TrainInputData => _trainInputData;

        public IReadOnlyList<float[]> TrainOutputData => _trainOutputData;

        public IReadOnlyList<float[]> TestInputData => _testInputData;

        public IReadOnlyList<float[]> TestOutputData => _testOutputData;

        public IReadOnlyList<int> TestOutputLabel => _testOutputLabel;

        private void LoadSample(string labelPath, string imagePath, ICollection<float[]> input, ICollection<float[]> output, ICollection<int> labels)
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
                input.Add(singleData.ImageData);
                output.Add(singleData.LabelArray);
                labels?.Add(label);
            }
        }

        public IEnumerator DownloadData()
        {
            var directory = $"{Application.temporaryCachePath}/mnist";
            Directory.CreateDirectory(directory);
            yield return DownloadGZip("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
                , directory,
                "train-labels-idx1-ubyte");
            yield return DownloadGZip("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
                , directory,
                "train-images-idx3-ubyte");
            yield return DownloadGZip("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
                , directory,
                "t10k-images-idx3-ubyte");
            yield return DownloadGZip("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
                , directory,
                "t10k-labels-idx1-ubyte");
            LoadSample($"{directory}/train-labels-idx1-ubyte"
                , $"{directory}/train-images-idx3-ubyte"
                , _trainInputData
                , _trainOutputData
                , null);
            LoadSample($"{directory}/t10k-labels-idx1-ubyte"
                , $"{directory}/t10k-images-idx3-ubyte"
                , _testInputData
                , _testOutputData
                , _testOutputLabel);
        }

        private static IEnumerator DownloadGZip(string url, string directory, string fileName)
        {
            if (File.Exists($"{directory}/{fileName}")) yield break;
            Debug.Log($"downloading {url}");
            var webRequest = UnityWebRequest.Get(url);
            yield return webRequest.SendWebRequest();
            if (webRequest.result == UnityWebRequest.Result.Success)
            {
                using var decompressedFileStream = File.Create($"{directory}/{fileName}");
                using var decompressionStream = new GZipStream(new MemoryStream(webRequest.downloadHandler.data), CompressionMode.Decompress);
                decompressionStream.CopyTo(decompressedFileStream);
            }
            else
            {
                Debug.LogError(webRequest.error);
            }
        }

    }

}