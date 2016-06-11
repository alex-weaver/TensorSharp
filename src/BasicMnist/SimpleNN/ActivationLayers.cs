using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public abstract class ActivationLayer : Layer
    {
        protected readonly Tensor activation, gradInput;

        public ActivationLayer(IAllocator allocator, DType elementType, params long[] shape)
        {
            this.activation = new Tensor(allocator, elementType, shape);
            this.gradInput = new Tensor(allocator, elementType, shape);
        }

        public override Tensor Output { get { return activation; } }
        public override Tensor GradInput { get { return gradInput; } }

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
            // no parameters in activation layers
        }
    }

    public class SigmoidLayer : ActivationLayer
    {
        public SigmoidLayer(IAllocator allocator, DType elementType, params long[] shape)
            : base(allocator, elementType, shape)
        {
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            Ops.Sigmoid(activation, input);
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            UpdateGradInput(gradOutput, activation);
            return gradInput;
        }

        private void UpdateGradInput(Tensor gradOutput, Tensor output)
        {
            // Computes  gradInput = gradOutput .* (1 - output) .* output

            gradOutput.TVar()
                .CMul(1 - output.TVar())
                .CMul(output)
                .Evaluate(gradInput);

       }

    }

    /// <summary>
    /// Output element x' -> x if x > threshold; val otherwise
    /// </summary>
    public class ThresholdLayer : ActivationLayer
    {
        private readonly float threshold, val;

        public ThresholdLayer(IAllocator allocator, DType elementType, long[] shape, float threshold, float val)
            : base(allocator, elementType, shape)
        {
            this.threshold = threshold;
            this.val = val;
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            var keepElements = input.TVar() > threshold;
            (input.TVar().CMul(keepElements) + (1 - keepElements) * val)
                .Evaluate(activation);
            
            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            UpdateGradInput(input, gradOutput);
            return gradInput;
        }

        private void UpdateGradInput(Tensor input, Tensor gradOutput)
        {
            // Retains gradients only where input x > threshold

            gradOutput.TVar().CMul(input.TVar() > threshold)
                .Evaluate(gradInput);
        }
    }

    public class ReLULayer : ThresholdLayer
    {
        public ReLULayer(IAllocator allocator, DType elementType, params long[] shape)
            : base(allocator, elementType, shape, 0, 0)
        {
        }
    }
}
