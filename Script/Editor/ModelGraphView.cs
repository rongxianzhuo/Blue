using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;
using UnityEngine;
using UnityEngine.UIElements;

namespace Blue.Editor
{

    public class ModelGraphView : GraphView
    {

        private readonly OutputNode _outputNode = new OutputNode();

        public ModelGraphView()
        {
            SetupZoom(ContentZoomer.DefaultMinScale, ContentZoomer.DefaultMaxScale);
            this.AddManipulator(new ContentDragger());
            this.AddManipulator(new SelectionDragger());
            this.AddManipulator(new RectangleSelector());
            this.AddManipulator(new FreehandSelector());
            AddElement(_outputNode);
            var searchWindowProvider = ScriptableObject.CreateInstance<NodeSearchWindowProvider>();
            searchWindowProvider.Initialize(this);
            nodeCreationRequest += context =>
            {
                SearchWindow.Open(new SearchWindowContext(context.screenMousePosition), searchWindowProvider);
            };
        }

        public void ConnectPort(Port p1, Port p2)
        {
            AddElement(p1.ConnectTo(p2));
        }

        private void ForeachInputNode(BlueNode node, Action<BlueNode> action)
        {
            node.ForeachInputNode(n =>
            {
                action(n);
                ForeachInputNode(n, action);
            });
        }

        public BlueNode FindNode(string nodeName)
        {
            foreach (var node in nodes)
            {
                var n = node as BlueNode;
                if (n == null) continue;
                if (n.Name == nodeName) return n;
            }

            return null;
        }


        public void LoadModel(GraphAsset asset)
        {
            asset.ForeachNode((method, parameters, info) =>
            {
                BlueNode node = method switch
                {
                    "TensorNode" => new TensorNode(),
                    "DenseLayer" => new DenseLayer(),
                    _ => throw new Exception("Unknown error")
                };
                node.SetSaveInfo(this, parameters);
                node.SetDisplayInfo(info);
                AddElement(node);
            });
            ConnectPort(FindNode(asset.OutputNodeName).OutputPort, _outputNode.InputNode);
            _outputNode.SetDisplayInfo(asset.OutputNodeInfo);
        }

        public void Save(Stream stream)
        {
            var list = new List<BlueNode>();
            ForeachInputNode(_outputNode, node => list.Add(node));
            list.Reverse();
            var asset = new GraphAsset();
            foreach (var node in list)
            {
                node.GetSaveInfo(out var method, out var parameters);
                asset.AddNode(node.Name, method, parameters, node.GetDisplayInfo());
            }
            asset.SaveToStream(stream, _outputNode.GetDisplayInfo());
        }
        
        public override List<Port> GetCompatiblePorts(Port startAnchor, NodeAdapter nodeAdapter)
        {
            var compatiblePorts = new List<Port>();
            foreach (var port in ports.ToList())
            {
                if (startAnchor.node == port.node ||
                    startAnchor.direction == port.direction ||
                    startAnchor.portType != port.portType)
                {
                    continue;
                }
 
                compatiblePorts.Add(port);
            }
            return compatiblePorts;
        }
        
    }

}