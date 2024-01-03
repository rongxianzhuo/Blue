using System;
using System.Linq;
using Blue.Kit;
using UnityEditor.Experimental.GraphView;
using UnityEngine.UIElements;

namespace Blue.Editor
{
    public sealed class DenseLayer : BlueNode
    {
        
        public readonly Port Input;
        public readonly Port Output;
        
        private readonly TextField _sizeField = new TextField();
        private readonly TextField _activationField = new TextField();
        
        public override Port OutputPort => Output;

        public int Size => int.Parse(_sizeField.value);

        public string Activation => _activationField.value;

        public BlueNode InputNode
        {
            get
            {
                var node = Input.connections.FirstOrDefault();
                return node?.output.node as BlueNode;
            }
        }

        public DenseLayer()
        {
            title = "DenseLayer";
            
            Input = Port.Create<Edge>(Orientation.Horizontal, Direction.Input, Port.Capacity.Single, typeof(Port));
            Input.portName = "Input";
            inputContainer.Add(Input);
            
            Output = Port.Create<Edge>(Orientation.Horizontal, Direction.Output, Port.Capacity.Single, typeof(Port));
            Output.portName = "Output";
            outputContainer.Add(Output);
            
            mainContainer.Add(_sizeField);
            mainContainer.Add(_activationField);
            _sizeField.value = "0";
        }

        public override void SetSaveInfo(ModelGraphView graphView, object[] parameters)
        {
            Name = parameters[0].ToString();
            graphView.ConnectPort(graphView.FindNode(parameters[1].ToString()).OutputPort, Input);
            _sizeField.value = parameters[2].ToString();
            _activationField.value = parameters[3].ToString();
        }

        public override void ForeachInputNode(Action<BlueNode> action)
        {
            action(InputNode);
        }

        public override void GetSaveInfo(out string method, out object[] parameters)
        {
            method = "DenseLayer";
            parameters = new object[] { Name, InputNode.Name, Size, Activation };
        }
    }
}