using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class LinearLayer : Layer
    {
        private Tensor weights, bias, activation, gradInput;
        private Tensor gradWeights, gradBias;

        private readonly int batchSize, nOutput;

        public LinearLayer(IAllocator allocator, SeedSource seedSource, DType elementType, int nInput, int nOutput, int batchSize)
        {
            this.batchSize = batchSize;
            this.nOutput = nOutput;

            this.weights = new Tensor(allocator, elementType, nInput, nOutput);
            this.bias = new Tensor(allocator, elementType, 1, nOutput);

            this.activation = new Tensor(allocator, elementType, batchSize, nOutput);

            this.gradInput = new Tensor(allocator, elementType, batchSize, nInput);
            this.gradWeights = new Tensor(allocator, elementType, nInput, nOutput);
            this.gradBias = new Tensor(allocator, elementType, 1, nOutput);

            InitWeightsLinear(seedSource, weights, bias);
        }

        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }

        public override IEnumerable<Tensor> GetParameters()
        {
            yield return weights;
            yield return bias;
        }

        public override IEnumerable<Tensor> GetGradParameters()
        {
            yield return gradWeights;
            yield return gradBias;
        }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            var weightSize = weights.ElementCount();
            var biasSize = bias.ElementCount();

            weights.TVar().View(weightSize)
                .Evaluate(parameters.TVar().Narrow(0, 0, weightSize));

            bias.TVar().View(biasSize)
                .Evaluate(parameters.TVar().Narrow(0, weightSize, biasSize));

            gradWeights.TVar().View(weightSize)
                .Evaluate(gradParameters.TVar().Narrow(0, 0, weightSize));

            gradBias.TVar().View(biasSize)
                .Evaluate(gradParameters.TVar().Narrow(0, weightSize, biasSize));
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            // activation = [bias] + input * weights
            // where [bias] means broadcast the bias vector
            bias.TVar().Expand(batchSize, nOutput)
                .Addmm(1, 1, input, weights)
                .Evaluate(activation);

            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            UpdateGradInput(gradOutput);
            AccWeightGrads(input, gradOutput);
            return gradInput;
        }

        private void AccWeightGrads(Tensor input, Tensor gradOutput)
        {
            gradWeights.TVar().Addmm(1, 1, input.TVar().Transpose(), gradOutput)
                .Evaluate(gradWeights);

            (gradBias + gradOutput.TVar().Sum(0))
                .Evaluate(gradBias);
        }

        private void UpdateGradInput(Tensor gradOutput)
        {
            gradOutput.TVar().Dot(weights.TVar().Transpose())
                .Evaluate(gradInput);
        }

        private void InitWeightsLinear(SeedSource seedSource, Tensor weights, Tensor bias)
        {
            var stdv = 1.0f / (float)Math.Sqrt(weights.Sizes[1]);
            Ops.RandomUniform(weights, seedSource, -stdv, stdv);
            Ops.RandomUniform(bias, seedSource, -stdv, stdv);
        }
    }
}
