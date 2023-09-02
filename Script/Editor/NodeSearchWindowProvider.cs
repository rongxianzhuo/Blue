using System;
using System.Collections.Generic;
using UnityEditor.Experimental.GraphView;
using UnityEngine;

namespace Blue.Editor
{
    public class NodeSearchWindowProvider : ScriptableObject, ISearchWindowProvider
    {

        private readonly List<SearchTreeEntry> _searchTree = new List<SearchTreeEntry>();

        private ModelGraphView _graphView;
 
        public void Initialize(ModelGraphView graphView)
        {
            _graphView = graphView;
        }
 
        List<SearchTreeEntry> ISearchWindowProvider.CreateSearchTree(SearchWindowContext context)
        {
            if (_searchTree.Count > 0) return _searchTree;
            _searchTree.Add(new SearchTreeGroupEntry(new GUIContent("Create Node")));
 
            foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            {
                foreach (var type in assembly.GetTypes())
                {
                    if (type.IsSubclassOf(typeof(BlueNode)))
                    {
                        _searchTree.Add(new SearchTreeEntry(new GUIContent(type.Name)) { level = 1, userData = type });
                    }
                }
            }
 
            return _searchTree;
        }
 
        bool ISearchWindowProvider.OnSelectEntry(SearchTreeEntry searchTreeEntry, SearchWindowContext context)
        {
            var node = Activator.CreateInstance((Type) searchTreeEntry.userData) as BlueNode;
            _graphView.AddElement(node);
            return true;
        }
    }
}