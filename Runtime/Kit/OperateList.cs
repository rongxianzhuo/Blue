using System;
using System.Collections.Generic;
using Blue.Core;

namespace Blue.Kit
{
    public class OperateList : IDisposable
    {

        private readonly List<Operate> _list = new List<Operate>();

        public OperateList Add(Operate operate)
        {
            _list.Add(operate);
            return this;
        }

        public OperateList Dispatch()
        {
            foreach (var o in _list)
            {
                o.Dispatch();
            }
            return this;
        }

        public void Dispose()
        {
            foreach (var o in _list)
            {
                o.Dispose();
            }
            _list.Clear();
        }
    }
}