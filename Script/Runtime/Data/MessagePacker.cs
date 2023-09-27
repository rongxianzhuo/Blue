using System;
using System.Collections.Generic;
using System.IO;

namespace Blue.Data
{
    public class MessagePacker
    {
        
        private const int SizeOfFloat = sizeof(float);

        private readonly Stream _stream;
        private readonly byte[] _byte4 = new byte[4];

        public MessagePacker(Stream stream)
        {
            _stream = stream;
        }

        public byte UnpackByte()
        {
            return (byte) _stream.ReadByte();
        }

        public byte[] UnpackByteArray(int length)
        {
            var array = new byte[length];
            for (var i = 0; i < length; i++)
            {
                array[i] = UnpackByte();
            }

            return array;
        }

        public int UnpackInt32()
        {
            ReadBytes(_byte4);
            return BitConverter.ToInt32(_byte4);
        }

        public int[] UnpackInt32Array(int length)
        {
            var array = new int[length];
            for (var i = 0; i < length; i++)
            {
                array[i] = UnpackInt32();
            }

            return array;
        }

        public float UnpackSingle()
        {
            ReadBytes(_byte4);
            return BitConverter.ToSingle(_byte4);
        }

        public float[] UnpackSingleArray(int length)
        {
            var array = new float[length];
            for (var i = 0; i < length; i++)
            {
                array[i] = UnpackSingle();
            }

            return array;
        }

        public void Pack(float f)
        {
            if (!BitConverter.TryWriteBytes(_byte4, f))
            {
                throw new Exception("Unknown error");
            }
            WriteBytes(_byte4);
        }

        public void Pack(IEnumerable<float> array)
        {
            foreach (var f in array)
            {
                Pack(f);
            }
        }

        private void ReadBytes(IReadOnlyCollection<byte> bytes)
        {
            for (var j = 0; j < bytes.Count; j++)
            {
                _byte4[j] = (byte) _stream.ReadByte();
            }
        }

        private void WriteBytes(IReadOnlyCollection<byte> bytes)
        {
            for (var i = 0; i < bytes.Count; i++)
            {
                _stream.WriteByte(_byte4[i]);
            }
        }

    }
}