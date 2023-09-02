using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Blue.Kit;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Blue.Editor
{

    public class ModelWindow : EditorWindow
    {

        private ModelGraphView _graphView;

        private void OnEnable()
        {
            if (_graphView != null) return;
            _graphView = new ModelGraphView();
            _graphView.StretchToParentSize();
            rootVisualElement.Add(_graphView);
            rootVisualElement.Add(new Button(() =>
            {
                var path = $"{Application.dataPath}/Blue/Demo/ModelAsset.bytes";
                using var stream = File.OpenWrite(path);
                _graphView.Save(stream);
                stream.Close();
            }) { text = "Save" });
            if (File.Exists($"{Application.dataPath}/Blue/Demo/ModelAsset.bytes"))
            {
                var graphAsset = new GraphAsset();
                var path = $"{Application.dataPath}/Blue/Demo/ModelAsset.bytes";
                using var stream = File.OpenRead(path);
                graphAsset.LoadFromStream(stream);
                stream.Close();
                _graphView.LoadModel(graphAsset);
            }
        }

        [MenuItem("Blue/Model Editor")]
        public static void OpenWindow()
        {
            var window = GetWindow<ModelWindow>("ModelEditor");
            window.Show();
        }
        
    }

}