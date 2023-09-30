using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace Blue.Data
{
    public class MessagePacker
    {

        private readonly Stream _stream;
        private readonly byte[] _byte4 = new byte[4];

        public bool IsEnd => _stream.Position >= _stream.Length;

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

        public Vector3 UnpackVector3()
        {
            var x = UnpackSingle();
            var y = UnpackSingle();
            var z = UnpackSingle();
            return new Vector3(x, y, z);
        }

        public Quaternion UnpackQuaternion()
        {
            var x = UnpackSingle();
            var y = UnpackSingle();
            var z = UnpackSingle();
            var w = UnpackSingle();
            return new Quaternion(x, y, z, w);
        }

        public void Pack(Quaternion quaternion)
        {
            Pack(quaternion.x);
            Pack(quaternion.y);
            Pack(quaternion.z);
            Pack(quaternion.w);
        }

        public void Pack(Vector3 vector3)
        {
            Pack(vector3.x);
            Pack(vector3.y);
            Pack(vector3.z);
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