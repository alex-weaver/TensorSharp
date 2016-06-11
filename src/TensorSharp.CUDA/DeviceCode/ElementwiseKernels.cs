﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.DeviceCode
{
    [Precompile]
    public class ElementwiseKernels : CudaCode
    {
        public ElementwiseKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "Math", "ApplyMacros")
        {
        }

        private static string GetFullCode()
        {
            var result = new PermutationGenerator();
            AppendTTFunc(result, "abs", "fabs");
            AppendTTFunc(result, "neg", "-");
            AppendTTFunc(result, "sign", "sgn");

            AppendTTFunc(result, "sqrt", "sqrt");
            AppendTTFunc(result, "exp", "exp");
            AppendTTFunc(result, "log", "log");
            AppendTTFunc(result, "log1p", "log1p");
            AppendTTFunc(result, "floor", "floor");
            AppendTTFunc(result, "ceil", "ceil");
            AppendTTFunc(result, "round", "round");
            AppendTTFunc(result, "trunc", "trunc");
            AppendTTFunc(result, "frac", "Frac");

            AppendTTFunc(result, "sin", "sin");
            AppendTTFunc(result, "cos", "cos");
            AppendTTFunc(result, "tan", "tan");
            AppendTTFunc(result, "asin", "asin");
            AppendTTFunc(result, "acos", "acos");
            AppendTTFunc(result, "atan", "atan");
            AppendTTFunc(result, "sinh", "sinh");
            AppendTTFunc(result, "cosh", "cosh");
            AppendTTFunc(result, "tanh", "tanh");

            AppendTTFunc(result, "sigmoid", "Sigmoid");

            result.AddApplyTTT("atan2", "*a = atan2f(*b, *c);");

            result.AddApplyTS("t1_pow", "*a = powf(*a, b);");
            result.AddApplyTTS("t2_pow", "*a = powf(*b, c);");
            result.AddApplyTS("t1_tpow", "*a = powf(b, *a);");
            result.AddApplyTTS("t2_tpow", "*a = powf(c, *b);");

            result.AddApplyTTTS("lerp", "*a = Lerp(*b, *c, d);");

            result.AddApplyTSS("t1_clamp", "*a = Clamp(*a, b, c);");
            result.AddApplyTTSS("t2_clamp", "*a = Clamp(*b, c, d);");

            AppendTTSFunc(result, "add", "add_op");
            AppendTTSFunc(result, "sub", "sub_op");
            AppendTTSFunc(result, "rsub", "rsub_op");
            AppendTTSFunc(result, "mul", "mul_op");
            AppendTTSFunc(result, "div", "div_op");
            AppendTTSFunc(result, "rdiv", "rdiv_op");
            AppendTTSFunc(result, "mod", "Mod_op");

            AppendTTSFunc(result, "gt", "gt_op");
            AppendTTSFunc(result, "lt", "lt_op");
            AppendTTSFunc(result, "ge", "gt_op");
            AppendTTSFunc(result, "le", "le_op");
            AppendTTSFunc(result, "eq", "eq_op");
            AppendTTSFunc(result, "ne", "ne_op");

            AppendTTTFunc(result, "cadd", "add_op");
            AppendTTTFunc(result, "csub", "sub_op");
            AppendTTTFunc(result, "cmul", "mul_op");
            AppendTTTFunc(result, "cdiv", "div_op");
            AppendTTTFunc(result, "cmod", "Mod_op");

            AppendTTTFunc(result, "cgt", "gt_op");
            AppendTTTFunc(result, "clt", "lt_op");
            AppendTTTFunc(result, "cge", "gt_op");
            AppendTTTFunc(result, "cle", "le_op");
            AppendTTTFunc(result, "ceq", "eq_op");
            AppendTTTFunc(result, "cne", "ne_op");


            return result.ToString();
        }

        private static void AppendTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyT("t1_" + kernelBaseName, string.Format("*v = {0}(*v);", func));
            pg.AddApplyTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b);", func));
        }

        private static void AppendTTSFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTS("t1_" + kernelBaseName, string.Format("*a = {0}(*a, b);", func));
            pg.AddApplyTTS("t2_" + kernelBaseName, string.Format("*a = {0}(*b, c);", func));
        }

        private static void AppendTTTFunc(PermutationGenerator pg, string kernelBaseName, string func)
        {
            pg.AddApplyTT("t1_" + kernelBaseName, string.Format("*a = {0}(*a, *b);", func));
            pg.AddApplyTTT("t2_" + kernelBaseName, string.Format("*a = {0}(*b, *c);", func));
        }
    }
}
