using System;
using System.Collections.Generic;
using Blue.Core;
using Blue.Graph;

namespace Blue.Optimizers
{
    public interface IOptimizer : IDisposable
    {
        void Step();
    }
}