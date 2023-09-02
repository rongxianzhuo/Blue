using System;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;

namespace Blue.Editor
{
    public abstract class BlueNode : Node
    {

        private readonly long _id = (long) (DateTime.Now - DateTime.UnixEpoch).TotalMilliseconds;

        public string Name { get; protected set; } = ((long)(DateTime.Now - DateTime.UnixEpoch).TotalMilliseconds).ToString();

        public abstract Port OutputPort { get; }

        public abstract void SetSaveInfo(ModelGraphView graphView, object[] parameters, GraphAsset.NodeInfo info);

        public abstract void ForeachInputNode(Action<BlueNode> action);

        public abstract void GetSaveInfo(out string method, out object[] parameters);

        public GraphAsset.NodeInfo GetInfo()
        {
            return new GraphAsset.NodeInfo()
            {
                position = GetPosition()
            };
        }

    }
}