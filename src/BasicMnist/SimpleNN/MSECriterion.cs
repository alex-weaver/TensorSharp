using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class MSECriterion : ICriterion
    {
        private Tensor output;
        private Tensor gradInput;


        public MSECriterion(IAllocator allocator, int batchSize, int outputSize)
        {
            this.output = new Tensor(allocator, DType.Float32, 1);
            this.gradInput = new Tensor(allocator, DType.Float32, batchSize, outputSize);
        }

        public Tensor UpdateOutput(Tensor input, Tensor target)
        {
            (input.TVar() - target)
                .Pow(2)
                .MeanAll()
                .Evaluate(output);

            return output;
        }

        public Tensor UpdateGradInput(Tensor input, Tensor target)
        {
            var norm = 2.0f / input.ElementCount();

            ((input.TVar() - target) * norm)
                .Evaluate(gradInput);

            return gradInput;
        }
    }
}
