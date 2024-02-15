using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Blue.Graph
{
    public class ComputationalGraph
    {

        public readonly ComputationalNode Output;

        private readonly Dictionary<int, ComputationalNode> _parameterNodes = new Dictionary<int, ComputationalNode>();
        private readonly List<ComputationalNode> _nodes = new List<ComputationalNode>();

        public IEnumerable<ComputationalNode> ParameterNodes => _parameterNodes.Values;

        public ComputationalGraph(ComputationalNode output)
        {
            Output = output;
            AddNode(output);
        }

        public void LoadParameterFile(string dirPath)
        {
            foreach (var pair in _parameterNodes)
            {
                if (File.Exists(Path.Combine(dirPath, $"{pair.Key}.bytes")))
                {
                    using var stream = File.OpenRead(Path.Combine(dirPath, $"{pair.Key}.bytes"));
                    pair.Value.LoadFromStream(stream);
                    stream.Close();
                }
                else
                {
                    Debug.LogWarning($"No parameter file: {pair.Key}");
                }
            }
        }

        public void SaveParameterFile(string dirPath)
        {
            Directory.CreateDirectory(dirPath);
            foreach (var pair in _parameterNodes)
            {
                using var stream = File.OpenWrite(Path.Combine(dirPath, $"{pair.Key}.bytes"));
                pair.Value.SaveToStream(stream);
                stream.Close();
            }
        }

        private void AddNode(ComputationalNode node)
        {
            if (_nodes.Contains(node)) return;
            if (node.IsParameter) _parameterNodes.Add(_parameterNodes.Count + 1, node);
            if (node.InputNodes.Count == 0)
            {
                _nodes.Insert(0, node);
                return;
            }

            var i = _nodes.Count - 1;
            while (i >= 0)
            {
                if (node.InputNodes.Contains(_nodes[i])) break;
                i--;
            }
            _nodes.Insert(i + 1, node);
            foreach (var n in node.InputNodes)
            {
                AddNode(n);
            }
        }

        public void Forward()
        {
            foreach (var node in _nodes)
            {
                node.Forward();
            }
        }

        public void Backward()
        {
            for (var i = _nodes.Count - 1; i >= 0; i--)
            {
                _nodes[i].Backward();
            }
        }

        public void ClearGradient()
        {
            foreach (var node in _nodes)
            {
                node.ClearGradient();
            }
        }

        public void CopyParameterTo(ComputationalGraph other)
        {
            foreach (var pair in _parameterNodes)
            {
                other._parameterNodes[pair.Key].SetData(pair.Value.InternalSync());
            }
        }

        public void DisposeNodes()
        {
            foreach (var n in _nodes)
            {
                n.Dispose();
            }
            _nodes.Clear();
            _parameterNodes.Clear();
        }
    }
}