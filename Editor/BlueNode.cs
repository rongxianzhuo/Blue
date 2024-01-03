using System;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;

namespace Blue.Editor
{
    public abstract class BlueNode : Node
    {

        public string Name { get; protected set; } = ((long)(DateTime.Now - DateTime.UnixEpoch).TotalMilliseconds).ToString();

        public abstract Port OutputPort { get; }

        public abstract void SetSaveInfo(ModelGraphView graphView, object[] parameters);

        public abstract void ForeachInputNode(Action<BlueNode> action);

        public abstract void GetSaveInfo(out string method, out object[] parameters);

        public GraphAsset.NodeDisplayInfo GetDisplayInfo()
        {
            return new GraphAsset.NodeDisplayInfo()
            {
                position = GetPosition()
            };
        }

        public void SetDisplayInfo(GraphAsset.NodeDisplayInfo info)
        {
            SetPosition(info.position);
        }

    }
}