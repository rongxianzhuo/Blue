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
        
        public override ComputationalNode Forward(params ComputationalNode[] input)
        {
            var x = _modules[0].Forward(input);
            for (var i = 1; i < _modules.Length; i++)
            {
                x = _modules[i].Forward(x);
            }
            return x;
        }
    }
}