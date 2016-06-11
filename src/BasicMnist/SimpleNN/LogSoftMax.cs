using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.CUDA;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class LogSoftMax : Layer
    {
        private readonly Tensor activation, gradInput;

        public LogSoftMax(IAllocator allocator, DType elementType, int nInput, int batchSize)
        {
            this.activation = new Tensor(allocator, elementType, batchSize, nInput);
            this.gradInput = new Tensor(allocator, elementType, batchSize, nInput);
        }


        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }
        public override IEnumerable<Tensor> GetGradParameters() { return Enumerable.Empty<Tensor>(); }
        public override IEnumerable<Tensor> GetParameters() { return Enumerable.Empty<Tensor>(); }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            // no parameters
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            var maxes = input.TVar().Max(1);
            var maxesExp = maxes.Expand(input.Sizes);

            var d = (input - maxesExp).Exp().Sum(1).Log();
            var logSum = (d + maxes).Expand(input.Sizes);

            (input - logSum)
                .Evaluate(activation);

            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            var go = gradOutput.TVar();
            var a = activation.TVar().Exp().CMul(go.Sum(1).Expand(activation.Sizes));
            (go - a)
                .Evaluate(gradInput);

            return gradInput;
        }
    }

    public class LogSoftMaxDNN : Layer
    {
        private readonly Tensor activation, gradInput;

        public LogSoftMaxDNN(IAllocator allocator, DType elementType, int nInput, int batchSize)
        {
            this.activation = new Tensor(allocator, elementType, batchSize, nInput);
            this.gradInput = new Tensor(allocator, elementType, batchSize, nInput);
        }


        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }
        public override IEnumerable<Tensor> GetGradParameters() { return Enumerable.Empty<Tensor>(); }
        public override IEnumerable<Tensor> GetParameters() { return Enumerable.Empty<Tensor>(); }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            // no parameters
        }

        private Tensor As4d(Tensor value)
        {
            return value.View(value.Sizes[0], value.Sizes[1], 1, 1);
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            using (var input4d = As4d(input))
            using (var activation4d = As4d(activation))
            {
                DNN.SoftmaxForward(DNNSoftmaxAlgorithm.Log, DNNSoftmaxMode.Instance, input4d, activation4d);
            }

            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            using (var activation4d = As4d(activation))
            using (var gradInput4d = As4d(gradInput))
            using (var gradOutput4d = As4d(gradOutput))
            {
                DNN.SoftmaxBackward(DNNSoftmaxAlgorithm.Log, DNNSoftmaxMode.Instance, activation4d, gradInput4d, gradOutput4d);
            }

            return gradInput;
        }

    }
}
