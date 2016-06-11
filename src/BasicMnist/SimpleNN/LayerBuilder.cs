using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace BasicMnist.SimpleNN
{
    /// <summary>
    /// Chooses between different implementations of the same layer depending on the targeted platform
    /// </summary>
    public static class LayerBuilder
    {
        public static Layer BuildLogSoftMax(IAllocator allocator, DType elementType, int batchSize, int nInputs, bool useCudnn = false)
        {
            if (allocator is CpuAllocator)
            {
                return new LogSoftMax(allocator, elementType, nInputs, batchSize);
            }
            else if (allocator is CudaAllocator)
            {
                if (useCudnn)
                {
                    return new LogSoftMaxDNN(allocator, elementType, nInputs, batchSize);
                }
                else
                {
                    return new LogSoftMax(allocator, elementType, nInputs, batchSize);
                }
            }
            else
            {
                throw new NotSupportedException("Allocator type " + allocator.GetType() + " not supported");
            }
        }

        public static Conv2Layer BuildConvLayer(IAllocator allocator, SeedSource seedSource, DType elementType, int batchSize, int inputWidth, int inputHeight, int nInputPlane, int nOutputPlane, ConvolutionDesc2d cd, bool useCudnn = false)
        {
            if (allocator is CpuAllocator)
            {
                return new Conv2Cpu(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd);
            }
            else if (allocator is CudaAllocator)
            {
                if (useCudnn)
                    return new Conv2Cudnn(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd);
                else
                    return new Conv2Cuda(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd);
            }
            else
            {
                throw new NotSupportedException("Allocator type " + allocator.GetType() + " not supported");
            }
        }


        public static MaxPoolLayer BuildPoolLayer(IAllocator allocator, DType elementType, long[] inputSizes, ConvolutionDesc2d cdPool, bool useCudnn = false)
        {
            if (allocator is CpuAllocator)
            {
                return new MaxPoolCpu(allocator, elementType, (int)inputSizes[0], inputSizes[1], inputSizes[3], inputSizes[2], cdPool);
            }
            else if(allocator is CudaAllocator)
            {
                if (useCudnn)
                    return new MaxPoolCudnn(allocator, elementType, (int)inputSizes[0], inputSizes[1], inputSizes[3], inputSizes[2], cdPool);
                else
                    return new MaxPoolCuda(allocator, elementType, (int)inputSizes[0], inputSizes[1], inputSizes[3], inputSizes[2], cdPool);
            }
            else
            {
                throw new NotSupportedException("Allocator type " + allocator.GetType() + " not supported");
            }
        }
    }
}
