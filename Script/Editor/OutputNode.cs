using System;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;

namespace Blue.Editor
{
    public sealed class OutputNode : BlueNode
    {
        
        public readonly Port InputNode;

        public OutputNode()
        {
            title = "ModelOutput";
            InputNode = Port.Create<Edge>(Orientation.Horizontal, Direction.Input, Port.Capacity.Single, typeof(Port));
            InputNode.portName = "Input";
            inputContainer.Add(InputNode);
        }

        public override Port OutputPort => null;

        public override void SetSaveInfo(ModelGraphView graphAsset, object[] parameters, GraphAsset.NodeInfo info)
        {
            throw new NotImplementedException();
        }

        public override void ForeachInputNode(Action<BlueNode> action)
        {
            foreach (var edge in InputNode.connections)
            {
                action((BlueNode) edge.output.node);
            }
        }

        public override void GetSaveInfo(out string method, out object[] parameters)
        {
            throw new NotImplementedException();
        }
    }
}