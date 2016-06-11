using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public class ClassNLLCriterion : ICriterion
    {
        private readonly IAllocator allocator;
        private Tensor output;
        private Tensor gradInput;


        public ClassNLLCriterion(IAllocator allocator, int batchSize, int nClasses)
        {
            this.allocator = allocator;

            this.output = new Tensor(allocator, DType.Float32, 1);
            this.gradInput = new Tensor(allocator, DType.Float32, batchSize, nClasses);
        }

        public Tensor UpdateOutput(Tensor input, Tensor target)
        {
            var indices = target.TVar().View(target.Sizes[0], 1);

            var loss = input.TVar()
                .Gather(1, indices)
                .SumAll()
                 * (-1.0f / target.Sizes[0]);

            loss.Evaluate(output);

            return output;
        }

        public Tensor UpdateGradInput(Tensor input, Tensor target)
        {
            var norm = -1.0f / input.Sizes[0];

            TVar.Fill(0, allocator, DType.Float32, gradInput.Sizes)
                .Evaluate(gradInput);
            
            var indices = target.TVar().View(target.Sizes[0], 1);
            
            gradInput.TVar()
                .ScatterFill(norm, 1, indices)
                .Evaluate(gradInput);
            

            return gradInput;
        }
    }
}
