using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.CUDA;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public abstract class Conv2Layer : Layer
    {
        protected readonly ConvolutionDesc2d cd;

        protected Tensor weight, bias;
        protected Tensor gradWeight, gradBias;
        protected Tensor gradInput;
        protected Tensor activation;

        protected long[] inputSizes, outputSizes;


        public Conv2Layer(IAllocator allocator, SeedSource seedSource, DType elementType, int batchSize, int inputWidth, int inputHeight, int nInputPlane, int nOutputPlane, ConvolutionDesc2d cd)
        {
            this.cd = cd;

            this.weight = new Tensor(allocator, elementType, nOutputPlane, nInputPlane * cd.kW * cd.kH);
            this.bias = new Tensor(allocator, elementType, nOutputPlane, 1);

            this.gradWeight = new Tensor(allocator, elementType, this.weight.Sizes);
            this.gradBias = new Tensor(allocator, elementType, this.bias.Sizes);

            inputSizes = new long[] { batchSize, nInputPlane, inputHeight, inputWidth };
            this.gradInput = new Tensor(allocator, elementType, inputSizes);

            outputSizes = SpatialConvolutionMM.OutputSize(inputSizes, weight.Sizes, cd);
            this.activation = new Tensor(allocator, elementType, outputSizes);



            this.OutputSizes = outputSizes;

            var stdv = 1.0f / (float)Math.Sqrt(cd.kW * cd.kH * nInputPlane);
            Ops.RandomUniform(weight, seedSource, -stdv, stdv);
            Ops.RandomUniform(bias, seedSource, -stdv, stdv);
        }

        public long[] OutputSizes { get; private set; }

        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }



        public override IEnumerable<Tensor> GetGradParameters()
        {
            yield return gradWeight;
            yield return gradBias;
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            yield return weight;
            yield return bias;
        }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            var weightSize = weight.ElementCount();
            var biasSize = bias.ElementCount();

            weight.TVar().View(weightSize)
                .Evaluate(parameters.TVar().Narrow(0, 0, weightSize));

            bias.TVar().View(biasSize)
                .Evaluate(parameters.TVar().Narrow(0, weightSize, biasSize));

            gradWeight.TVar().View(weightSize)
                .Evaluate(gradParameters.TVar().Narrow(0, 0, weightSize));

            gradBias.TVar().View(biasSize)
                .Evaluate(gradParameters.TVar().Narrow(0, weightSize, biasSize));
        }
    }


    public class Conv2Cpu : Conv2Layer
    {
        private readonly Tensor finput, fgradInput;


        public Conv2Cpu(IAllocator allocator, SeedSource seedSource, DType elementType, int batchSize, int inputWidth, int inputHeight, int nInputPlane, int nOutputPlane, ConvolutionDesc2d cd)
            : base(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd)
        {
            var finputSizes = SpatialConvolutionMM.FInputSize(inputSizes, outputSizes, cd);
            this.finput = new Tensor(allocator, elementType, finputSizes);
            this.fgradInput = new Tensor(allocator, elementType, finputSizes);
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            SpatialConvolutionMM.Conv2Forward(input, activation, weight, bias, finput, cd);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            SpatialConvolutionMM.Conv2BackwardInput(input, gradOutput, gradInput, weight, finput, fgradInput, cd);
            SpatialConvolutionMM.Conv2BackwardFilter(input, gradOutput, gradWeight, gradBias, finput, fgradInput, cd);
            return gradInput;
        }
    }

    public class Conv2Cuda : Conv2Layer
    {
        private readonly Tensor finput, fgradInput;

        private readonly TensorSharp.CUDA.SpatialConvolution conv = new TensorSharp.CUDA.SpatialConvolution();


        public Conv2Cuda(IAllocator allocator, SeedSource seedSource, DType elementType, int batchSize, int inputWidth, int inputHeight, int nInputPlane, int nOutputPlane, ConvolutionDesc2d cd)
            : base(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd)
        {
            var finputSizes = TensorSharp.CUDA.SpatialConvolution.FInputSize(inputSizes, outputSizes, cd);
            this.finput = new Tensor(allocator, elementType, finputSizes);
            this.fgradInput = new Tensor(allocator, elementType, finputSizes);
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            conv.Conv2Forward(input, activation, weight, bias, finput, cd);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            conv.Conv2BackwardInput(input, gradOutput, gradInput, weight, finput, fgradInput, cd);
            conv.Conv2BackwardFilter(input, gradOutput, gradWeight, gradBias, finput, fgradInput, cd);
            return gradInput;
        }
    }

    public class Conv2Cudnn : Conv2Layer
    {
        private readonly CudaStorage workspace;

        private readonly DNNConvolutionFwdAlgo fwdAlgo = DNNConvolutionFwdAlgo.GEMM;
        private readonly DNNConvolutionBwdFilterAlgo bwdFilterAlgo = DNNConvolutionBwdFilterAlgo.Algo0;
        private readonly DNNConvolutionBwdDataAlgo bwdDataAlgo = DNNConvolutionBwdDataAlgo.Algo0;


        public Conv2Cudnn(IAllocator allocator, SeedSource seedSource, DType elementType, int batchSize, int inputWidth, int inputHeight, int nInputPlane, int nOutputPlane, ConvolutionDesc2d cd)
            : base(allocator, seedSource, elementType, batchSize, inputWidth, inputHeight, nInputPlane, nOutputPlane, cd)
        {
            // Reshape weight and bias - CuDNN expects the dimensions to be structured slightly differently
            this.weight = ViewReplace(this.weight, nOutputPlane, nInputPlane, cd.kH, cd.kW);
            this.bias = ViewReplace(this.bias, 1, nOutputPlane, 1, 1);
            this.gradWeight = ViewReplace(this.gradWeight, this.weight.Sizes);
            this.gradBias = ViewReplace(this.gradBias, this.bias.Sizes);


            var fwdWorkspace = DNN.GetConvolutionForwardWorkspaceSize(allocator, fwdAlgo, cd,
                new TensorShape(elementType, new long[] { batchSize, nInputPlane, inputHeight, inputWidth }),
                new TensorShape(weight),
                new TensorShape(activation));

            var bwdFilterWorkspace = DNN.GetConvolutionBackwardFilterWorkspaceSize(allocator, bwdFilterAlgo, cd,
                new TensorShape(elementType, new long[] { batchSize, nInputPlane, inputHeight, inputWidth }),
                new TensorShape(activation),
                new TensorShape(weight));

            var bwdFilterInputWorkspace = DNN.GetConvolutionBackwardDataWorkspaceSize(allocator, bwdDataAlgo, cd,
                new TensorShape(weight),
                new TensorShape(activation),
                new TensorShape(elementType, new long[] { batchSize, nInputPlane, inputHeight, inputWidth }));

            var workspaceSize = Math.Max(Math.Max(fwdWorkspace, bwdFilterWorkspace), bwdFilterInputWorkspace);

            this.workspace = (CudaStorage)allocator.Allocate(DType.UInt8, workspaceSize);
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            DNN.ConvForward(fwdAlgo, cd, workspace, input, weight, activation);
            DNN.AddTensor(bias, activation); // dims of bias with size = 1 are automatically broadcast over other dimensions
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            DNN.ConvolutionBackwardData(bwdDataAlgo, cd, workspace, weight, gradOutput, gradInput);
            DNN.ConvolutionBackwardFilter(bwdFilterAlgo, cd, workspace, input, gradOutput, gradWeight);
            DNN.ConvolutionBackwardBias(cd, gradOutput, gradBias);
            return gradInput;
        }

        private static Tensor ViewReplace(Tensor old, params long[] sizes)
        {
            var result = old.View(sizes);
            old.Dispose();
            return result;
        }
    }
}
