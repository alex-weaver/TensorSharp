using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace BasicMnist.SimpleNN
{
    public enum ModelMode
    {
        Train,
        Evaluate,
    }

    public abstract class Layer
    {
        public abstract IEnumerable<Tensor> GetParameters();
        public abstract IEnumerable<Tensor> GetGradParameters();

        public long GetParameterCount()
        {
            return GetParameters().Aggregate(0L, (acc, item) => acc + item.ElementCount());
        }

        public abstract void FlattenParams(Tensor parameters, Tensor gradParameters);


        public abstract Tensor Forward(Tensor input, ModelMode mode);
        public abstract Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode);

        public abstract Tensor Output { get; }
        public abstract Tensor GradInput { get; }
    }
}
