using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace BasicMnist
{
    public class SimpleTimer : IDisposable
    {
        private readonly Stopwatch sw;
        private readonly string endedMessage;

        public SimpleTimer(string endedMessage)
        {
            this.endedMessage = endedMessage;
            this.sw = new Stopwatch();
            sw.Start();
        }

        public void Dispose()
        {
            sw.Stop();
            Console.WriteLine(string.Format(endedMessage, sw.Elapsed.TotalMilliseconds));
        }
    }
}
