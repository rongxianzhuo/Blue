using System;

namespace Blue.Optimizers
{
    public interface IOptimizer : IDisposable
    {
        void Step();
    }
}