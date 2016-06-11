﻿using ManagedCuda;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Used by TSCudaContext to maintain per-device state
    /// </summary>
    public class DeviceState : IDisposable
    {
        private const int ScratchSpacePerSMStream = 4 * sizeof(float);


        public readonly CudaContext CudaContext;
        public readonly CudaDeviceProperties DeviceInfo;

        public readonly ObjectPool<CudaBlas> BlasHandles;
        public readonly ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext> DnnHandles;

        public readonly IDeviceAllocator MemoryAllocator;
        public readonly ScratchSpace ScratchSpace;


        public DeviceState(int deviceId)
        {
            this.CudaContext = new CudaContext(deviceId);
            this.DeviceInfo = this.CudaContext.GetDeviceInfo();

            this.BlasHandles = new ObjectPool<CudaBlas>(1, () =>
            {
                CudaContext.SetCurrent();
                return new CudaBlas();
            },
                blas => blas.Dispose());

            this.DnnHandles = new ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext>(0, () =>
            {
                CudaContext.SetCurrent();
                return new ManagedCuda.CudaDNN.CudaDNNContext();
            },
                dnn => dnn.Dispose());

            this.MemoryAllocator = new PoolingDeviceAllocator(CudaContext);
            this.ScratchSpace = AllocScratchSpace(CudaContext, DeviceInfo);
        }

        public void Dispose()
        {
            BlasHandles.Dispose();
            CudaContext.Dispose();
            this.MemoryAllocator.Dispose();
        }

        private static ScratchSpace AllocScratchSpace(CudaContext context, CudaDeviceProperties deviceProps)
        {
            var size = ScratchSpacePerSMStream * deviceProps.MultiProcessorCount;
            var buffer = context.AllocateMemory(size);
            return new ScratchSpace() { size = size, buffer = buffer };
        }
    }
}
