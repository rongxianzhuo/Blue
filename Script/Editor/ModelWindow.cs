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

        private static string _savePath;

        private ModelGraphView _graphView;

        public string SavePath => string.IsNullOrEmpty(_savePath)
            ? $"{Application.dataPath}/ModelAsset.blue.bytes"
            : _savePath;

        private void OnEnable()
        {
            if (_graphView != null) return;
            _graphView = new ModelGraphView();
            _graphView.StretchToParentSize();
            rootVisualElement.Add(_graphView);
            rootVisualElement.Add(new Button(() =>
            {
                using var stream = File.OpenWrite(SavePath);
                _graphView.Save(stream);
                stream.Close();
                AssetDatabase.Refresh();
            }) { text = "Save" });
            if (File.Exists(SavePath))
            {
                var graphAsset = new GraphAsset();
                using var stream = File.OpenRead(SavePath);
                graphAsset.LoadFromStream(stream);
                stream.Close();
                _graphView.LoadModel(graphAsset);
            }
        }

        [MenuItem("Blue/ModelEditor")]
        public static void OpenWindow()
        {
            _savePath = null;
            var window = GetWindow<ModelWindow>("ModelEditor");
            window.Show();
        }

        [MenuItem("Assets/Blue/ModelEditor", validate = true)]
        public static bool CanOpenEditor()
        {
            if (Selection.activeObject == null) return false;
            var path = AssetDatabase.GetAssetPath(Selection.activeObject);
            return path.EndsWith(".blue.bytes");
        }

        [MenuItem("Assets/Blue/ModelEditor")]
        public static void OpenEditorWithFile()
        {
            var path = AssetDatabase.GetAssetPath(Selection.activeObject).Substring(7);
            _savePath = $"{Application.dataPath}/{path}";
            var window = GetWindow<ModelWindow>("ModelEditor");
            window.Show();
        }
        
    }

}