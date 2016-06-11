using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public struct OutputAndGrads
    {
        public Tensor output;
        public Tensor[] grads;
    }

    public delegate OutputAndGrads GradFunc(Tensor[] parameters);

    public struct SgdConfig
    {
        public float LearningRate;
        public float Momentum;
    }

    public class SgdOptimizer
    {
        private readonly SgdConfig config;

        private Tensor[] gradAcc;


        public SgdOptimizer(SgdConfig config)
        {
            this.config = config;
        }

        public void Reset()
        {
            gradAcc = null;
        }

        // Modifies parameters in place
        // returns model output
        public Tensor Update(GradFunc grad, Tensor[] parameters)
        {
            var outputAndGrads = grad(parameters);
            Tensor output = outputAndGrads.output;
            Tensor[] gradients = outputAndGrads.grads;
            
                if (gradAcc == null)
                {
                    gradAcc = gradients.Select(x =>
                    {
                        var result = new Tensor(x.Allocator, x.ElementType, x.Sizes);
                        Ops.Fill(result, 0);
                        return result;
                    }).ToArray();
                }

                // gradAcc = gradAcc * momentum - learningRate * gradients
                for (int i = 0; i < gradients.Length; ++i)
                {
                    Ops.Mul(gradAcc[i], gradAcc[i], config.Momentum);
                    using (var temp = Ops.Mul(null, gradients[i], -config.LearningRate))
                    {
                        Ops.Add(gradAcc[i], gradAcc[i], temp);
                    }
                }
            

            for (int i = 0; i < parameters.Length; ++i)
            {
                Ops.Add(parameters[i], parameters[i], gradAcc[i]);
            }

            return output;
        }
    }
}
