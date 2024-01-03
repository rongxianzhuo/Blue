using System;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;
using UnityEngine.UIElements;

namespace Blue.Editor
{

    public sealed class TensorNode : BlueNode
    {

        private readonly Port _output;
        private readonly TextField _sizeField = new TextField();

        private int _size;
        
        public override Port OutputPort => _output;

        public TensorNode()
        {
            title = "Tensor";
            _output = Port.Create<Edge>(Orientation.Horizontal, Direction.Output, Port.Capacity.Single, typeof(Port));
            _output.portName = "Data";
            outputContainer.Add(_output);
            mainContainer.Add(_sizeField);
            _sizeField.value = "0";
            _sizeField.RegisterValueChangedCallback(e =>
            {
                if (int.TryParse(_sizeField.value, out var number)) _size = number;
                else _sizeField.SetValueWithoutNotify(_size.ToString());
            });
        }

        public override void SetSaveInfo(ModelGraphView graphView, object[] parameters)
        {
            Name = parameters[0].ToString();
            _sizeField.value = parameters[1].ToString();
        }

        public override void ForeachInputNode(Action<BlueNode> action)
        {
            
        }

        public override void GetSaveInfo(out string method, out object[] parameters)
        {
            method = "TensorNode";
            parameters = new object[] { Name, _size, false };
        }
    }

}