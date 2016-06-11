using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class DropoutLayer : Layer
    {
        private readonly SeedSource seedSource;
        private readonly IAllocator allocator;
        private readonly DType elementType;

        private readonly Tensor activation, gradInput, noise;
        private readonly float pRemove;


        public DropoutLayer(IAllocator allocator, SeedSource seedSource, DType elementType, float pRemove, params long[] shape)
        {
            this.seedSource = seedSource;
            this.allocator = allocator;
            this.elementType = elementType;

            this.pRemove = pRemove;

            this.activation = new Tensor(allocator, elementType, shape);
            this.gradInput = new Tensor(allocator, elementType, shape);
            this.noise = new Tensor(allocator, elementType, shape);
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
            // no parameters
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            Ops.Copy(activation, input);

            if (mode == ModelMode.Train)
            {
                var p = 1 - pRemove;

                TVar.RandomBernoulli(seedSource, p, allocator, elementType, noise.Sizes)
                    .Div(p)
                    .Evaluate(noise);

                activation.TVar()
                    .CMul(noise)
                    .Evaluate(activation);
            }

            return activation;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {   
            UpdateGradInput(gradOutput, activation, mode == ModelMode.Train);
            return gradInput;
        }

        private void UpdateGradInput(Tensor gradOutput, Tensor output, bool train)
        {
            Ops.Copy(gradInput, gradOutput);

            if (train)
            {
                Ops.Mul(gradInput, gradInput, noise);
            }
        }
    }
}
