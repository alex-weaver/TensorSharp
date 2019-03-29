using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.Core
{
    internal static class MemoryCopier
    {
        public static unsafe void Copy(IntPtr destination, IntPtr source, ulong length)
        {
            Buffer.MemoryCopy(source.ToPointer(), destination.ToPointer(), length, length);
        }
    }
}
