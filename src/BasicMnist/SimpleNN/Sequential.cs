using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp;

namespace BasicMnist.SimpleNN
{
    public class Sequential : Layer
    {
        private readonly List<Layer> layers = new List<Layer>();

        private Tensor lastOutput, lastGradInput;

        public Sequential()
        {
        }

        public override Tensor Output { get { return lastOutput; } }
        public override Tensor GradInput { get { return lastGradInput; } }


        public void Add(Layer layer)
        {
            this.layers.Add(layer);
        }

        public override IEnumerable<Tensor> GetParameters()
        {
            foreach (var layer in layers)
            {
                foreach (var tensor in layer.GetParameters())
                {
                    yield return tensor;
                }
            }
        }

        public override IEnumerable<Tensor> GetGradParameters()
        {
            foreach (var layer in layers)
            {
                foreach (var tensor in layer.GetGradParameters())
                {
                    yield return tensor;
                }
            }
        }

        public override void FlattenParams(Tensor parameters, Tensor gradParameters)
        {
            long offset = 0;
            for (int i = 0; i < layers.Count; i++)
            {
                var layer = layers[i];

                using (var paramSlice = parameters.Narrow(0, offset, layer.GetParameterCount()))
                using (var gradParamSlice = gradParameters.Narrow(0, offset, layer.GetParameterCount()))
                {
                    layer.FlattenParams(paramSlice, gradParamSlice);
                }

                offset += layer.GetParameterCount();
            }
        }

        public override Tensor Forward(Tensor input, ModelMode mode)
        {
            Tensor curOutput = input;
            foreach (var layer in layers)
            {
                curOutput = layer.Forward(curOutput, mode);
            }

            lastOutput = curOutput;
            return curOutput;
        }

        public override Tensor Backward(Tensor input, Tensor gradOutput, ModelMode mode)
        {
            var curGradOutput = gradOutput;

            for (int i = layers.Count - 1; i > 0; --i)
            {
                var layer = layers[i];
                var prevLayer = layers[i - 1];

                curGradOutput = layer.Backward(prevLayer.Output, curGradOutput, mode);
            }

            curGradOutput = layers[0].Backward(input, curGradOutput, mode);

            lastGradInput = curGradOutput;
            return curGradOutput;
        }

    }
}
