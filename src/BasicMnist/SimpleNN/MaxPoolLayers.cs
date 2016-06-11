using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;

namespace BasicMnist.SimpleNN
{
    public abstract class MaxPoolLayer : Layer
    {
        protected readonly ConvolutionDesc2d cd;
        protected readonly bool ceilMode;

        protected readonly Tensor activation, indices, gradInput;



        public MaxPoolLayer(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
        {
            this.cd = cd;
            this.ceilMode = ceilMode;

            var inputSizes = new long[] { batchSize, nInputPlane, inputWidth, inputHeight };
            var outputSizes = CpuMaxPoolingOps.OutputSize(inputSizes, ceilMode, cd);
            this.OutputSizes = outputSizes;

            this.activation = new Tensor(allocator, elementType, outputSizes);
            this.indices = new Tensor(allocator, elementType, outputSizes);
            this.gradInput = new Tensor(allocator, elementType, inputSizes);
        }

        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }


        public long[] OutputSizes { get; private set; }


        public override IEnumerable<Tensor> GetParameters()
        {
            return Enumerable.Empty<Tensor>();
        }

        public override IEnumerable<Tensor> GetGradParameters()
        {
            return Enumerable.Empty<Tensor>();
        }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            // no parameters
        }

    }



    public class MaxPoolCpu : MaxPoolLayer
    {
        public MaxPoolCpu(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {

        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            CpuMaxPoolingOps.SpatialMaxPoolingForward(input, activation, indices, cd, ceilMode);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            CpuMaxPoolingOps.SpatialMaxPoolingBackward(input, gradOutput, gradInput, indices, cd, ceilMode);
            return gradInput;
        }
    }

    public class MaxPoolCuda : MaxPoolLayer
    {
        private readonly TensorSharp.CUDA.DeviceCode.SpatialMaxPoolKernels maxPool = new TensorSharp.CUDA.DeviceCode.SpatialMaxPoolKernels();

        public MaxPoolCuda(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {

        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            maxPool.SpatialMaxPoolingForward(input, activation, indices, cd, ceilMode);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            maxPool.SpatialMaxPoolingBackward(input, gradOutput, gradInput, indices, cd, ceilMode);
            return gradInput;
        }
    }

    public class MaxPoolCudnn : MaxPoolLayer
    {
        private readonly DNNPoolingDesc poolingDesc;

        public MaxPoolCudnn(IAllocator allocator, DType elementType, int batchSize, long nInputPlane, long inputWidth, long inputHeight, ConvolutionDesc2d cd, bool ceilMode = true)
            : base(allocator, elementType, batchSize, nInputPlane, inputWidth, inputHeight, cd, ceilMode)
        {
            this.poolingDesc = new DNNPoolingDesc(DNNPoolingMode.Max, cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW);
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            DNN.PoolingForward(poolingDesc, input, activation);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            DNN.PoolingBackward(poolingDesc, input, activation, gradInput, gradOutput);
            return gradInput;
        }
    }
}
