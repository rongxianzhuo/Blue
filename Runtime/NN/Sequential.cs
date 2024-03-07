using Blue.Graph;

namespace Blue.Runtime.NN
{
    public class Sequential : Module
    {

        private readonly Module[] _modules;

        public Sequential(params Module[] modules)
        {
            _modules = modules;
            foreach (var module in modules)
            {
                RegisterModule(module);
            }
        }
        
        public override ComputationalGraph CreateGraph(params ComputationalNode[] input)
        {
            var x = _modules[0].CreateGraph(input).Output;
            for (var i = 1; i < _modules.Length; i++)
            {
                x = _modules[i].CreateGraph(x).Output;
            }
            return x.Graph();
        }
    }
}