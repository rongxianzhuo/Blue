using System;
using System.Collections.Generic;
using System.Linq;

namespace Blue.Graph
{
    public class ComputationalGraph : IDisposable
    {

        public readonly ComputationalNode Output;

        private readonly List<ComputationalNode> _nodes = new List<ComputationalNode>();

        public ComputationalGraph(ComputationalNode output, params ComputationalNode[] excludeInputs)
        {
            Output = output;
            AddNode(output, excludeInputs);
        }

        private void AddNode(ComputationalNode node, ComputationalNode[] excludeInputs)
        {
            if (_nodes.Contains(node)) return;
            if (excludeInputs != null && excludeInputs.Contains(node)) return;
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
                AddNode(n, excludeInputs);
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

        public void Dispose()
        {
            foreach (var node in _nodes)
            {
                if (node.IsParameter) continue;
                node.Dispose();
            }
        }
    }
}