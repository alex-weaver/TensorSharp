using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public class ViewLayer : Layer
    {
        private readonly long[] resultSize;

        private long[] lastInputSize;
        private Tensor activation, gradInput;

        public ViewLayer(params long[] resultSize)
        {
            this.resultSize = resultSize;
        }

        protected ViewLayer()
        {
        }

        public override Tensor GradInput { get { return gradInput; } }
        public override Tensor Output { get { return activation; } }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            if (gradInput != null)
                gradInput.Dispose();

            gradInput = gradOutput.View(lastInputSize);
            return gradInput;
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            if (activation != null)
                activation.Dispose();
            activation = input.View(resultSize);
            lastInputSize = input.Sizes;
            return activation;
        }

        public override IEnumerable<Tensor> GetGradParameters()
        {
            return Enumerable.Empty<Tensor>();
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            return Enumerable.Empty<Tensor>();
        }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            // no parameters
        }
    }
}
