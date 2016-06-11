using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public interface ICriterion
    {
        Tensor UpdateOutput(Tensor input, Tensor target);
        Tensor UpdateGradInput(Tensor input, Tensor target);
    }
}
