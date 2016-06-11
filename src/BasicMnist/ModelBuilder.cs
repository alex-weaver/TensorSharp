using BasicMnist.SimpleNN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;

namespace BasicMnist
{
    public static class ModelBuilder
    {
        // Constructs a network composed of two fully-connected sigmoid layers
        public static void BuildMLP(IAllocator allocator, SeedSource seedSource, int batchSize, bool useCudnn, out Sequential model, out ICriterion criterion, out bool outputIsClassIndices)
        {
            int inputSize = MnistParser.ImageSize * MnistParser.ImageSize;
            int hiddenSize = 100;
            int outputSize = MnistParser.LabelCount;

            var elementType = DType.Float32;

            model = new Sequential();
            model.Add(new ViewLayer(batchSize, inputSize));
            model.Add(new LinearLayer(allocator, seedSource, elementType, inputSize, hiddenSize, batchSize));
            model.Add(new SigmoidLayer(allocator, elementType, batchSize, hiddenSize));

            model.Add(new LinearLayer(allocator, seedSource, elementType, hiddenSize, outputSize, batchSize));
            model.Add(new SigmoidLayer(allocator, elementType, batchSize, outputSize));

            criterion = new MSECriterion(allocator, batchSize, outputSize);

            outputIsClassIndices = false; // output is class (pseudo-)probabilities, not class indices
        }


        // Constructs a network with two fully-connected layers; one sigmoid, one softmax
        public static void BuildMLPSoftmax(IAllocator allocator, SeedSource seedSource, int batchSize, bool useCudnn, out Sequential model, out ICriterion criterion, out bool outputIsClassIndices)
        {
            int inputSize = MnistParser.ImageSize * MnistParser.ImageSize;
            int hiddenSize = 100;
            int outputSize = MnistParser.LabelCount;

            var elementType = DType.Float32;

            model = new Sequential();
            model.Add(new ViewLayer(batchSize, inputSize));

            model.Add(new LinearLayer(allocator, seedSource, elementType, inputSize, hiddenSize, batchSize));
            model.Add(new SigmoidLayer(allocator, elementType, batchSize, hiddenSize));

            model.Add(new LinearLayer(allocator, seedSource, elementType, hiddenSize, outputSize, batchSize));
            model.Add(LayerBuilder.BuildLogSoftMax(allocator, elementType, batchSize, outputSize, useCudnn));
            
            criterion = new ClassNLLCriterion(allocator, batchSize, outputSize);
            outputIsClassIndices = true; // output of criterion is class indices
        }

        // Constructs a convolutional network with two convolutional layers,
        // two fully-connected layers, ReLU units and a softmax on the output.
        public static void BuildCnn(IAllocator allocator, SeedSource seedSource, int batchSize, bool useCudnn, out Sequential model, out ICriterion criterion, out bool outputIsClassIndices)
        {
            var inputWidth = MnistParser.ImageSize;
            var inputHeight = MnistParser.ImageSize;

            var elementType = DType.Float32;

            var inputDims = new long[] { batchSize, 1, inputHeight, inputWidth };

            model = new Sequential();
            model.Add(new ViewLayer(inputDims));

            var outSize = AddCnnLayer(allocator, seedSource, elementType, model, inputDims, 20, useCudnn);
            outSize = AddCnnLayer(allocator, seedSource, elementType, model, outSize, 40, useCudnn);

            var convOutSize = outSize[1] * outSize[2] * outSize[3];
            model.Add(new ViewLayer(batchSize, convOutSize));

            var hiddenSize = 1000;
            var outputSize = 10;

            model.Add(new DropoutLayer(allocator, seedSource, elementType, 0.5f, batchSize, convOutSize));
            model.Add(new LinearLayer(allocator, seedSource, elementType, (int)convOutSize, hiddenSize, batchSize));
            model.Add(new ReLULayer(allocator, elementType, batchSize, hiddenSize));

            model.Add(new DropoutLayer(allocator, seedSource, elementType, 0.5f, batchSize, hiddenSize));
            model.Add(new LinearLayer(allocator, seedSource, elementType, hiddenSize, outputSize, batchSize));
            model.Add(LayerBuilder.BuildLogSoftMax(allocator, elementType, batchSize, outputSize, useCudnn));

            criterion = new ClassNLLCriterion(allocator, batchSize, outputSize);
            outputIsClassIndices = true; // output of criterion is class indices
        }

        private static long[] AddCnnLayer(IAllocator allocator, SeedSource seedSource, DType elementType, Sequential model, long[] inputSizes, int nOutputPlane, bool useCudnn)
        {
            var conv = LayerBuilder.BuildConvLayer(allocator, seedSource, elementType, (int)inputSizes[0], (int)inputSizes[3], (int)inputSizes[2], (int)inputSizes[1], nOutputPlane,
                new ConvolutionDesc2d(5, 5, 1, 1, 0, 0), useCudnn);
            model.Add(conv);

            var cdPool = new ConvolutionDesc2d(2, 2, 1, 1, 0, 0);
            var poolLayer = LayerBuilder.BuildPoolLayer(allocator, elementType, conv.OutputSizes, cdPool, useCudnn);
            model.Add(poolLayer);

            model.Add(new ReLULayer(allocator, elementType, poolLayer.OutputSizes));

            return poolLayer.OutputSizes;
        }
    }
}
