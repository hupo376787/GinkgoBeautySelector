using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GinkgoBeautySelector
{
    /// <summary>
    /// 用于调用 InsightFace 的 genderage.onnx 的简单包装器。
    /// 注意：根据你使用的具体 model，可能需要调整通道顺序和归一化方式（见 Preprocess 方法注释）。
    /// </summary>
    public sealed class GenderAgePredictor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _inputW;
        private readonly int _inputH;

        /// <summary>
        /// 构造器
        /// </summary>
        /// <param name="modelPath">genderage.onnx 路径</param>
        /// <param name="useCuda">是否尝试启用 CUDA（需要 Microsoft.ML.OnnxRuntime.Gpu）</param>
        /// <param name="inputW">模型输入宽度（默认 112）</param>
        /// <param name="inputH">模型输入高度（默认 112）</param>
        public GenderAgePredictor(string modelPath, bool useCuda = false, int inputW = 112, int inputH = 112)
        {
            var options = new SessionOptions();
            if (useCuda)
            {
                try
                {
                    // 需要引用 Microsoft.ML.OnnxRuntime.Gpu
                    options.AppendExecutionProvider_CUDA();
                }
                catch
                {
                    // 若追加失败，退回 CPU
                }
            }
            _session = new InferenceSession(modelPath, options);

            // 取第一个输入名，默认单输入模型
            _inputName = _session.InputMetadata.Keys.First();
            _inputW = inputW;
            _inputH = inputH;
        }

        /// <summary>
        /// 对传入的 faceCrop（任意尺寸）进行推理。
        /// 返回 (femaleProbability [0..1], ageEstimated)。
        /// 如果模型输出结构与这里假设不一致，需根据实际输出做调整。
        /// </summary>
        public (float femaleProb, float age) Predict(SKBitmap faceCrop)
        {
            if (faceCrop == null) throw new ArgumentNullException(nameof(faceCrop));

            using var resized = faceCrop.Resize(new SKImageInfo(_inputW, _inputH), SKFilterQuality.Medium);
            // Prepare tensor NCHW, float32, 默认按 RGB 且归一化到 [0,1]
            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputH, _inputW });
            for (int y = 0; y < _inputH; y++)
            {
                for (int x = 0; x < _inputW; x++)
                {
                    var c = resized.GetPixel(x, y);
                    // 默认：RGB order, normalized by 255
                    tensor[0, 0, y, x] = c.Red / 255f;
                    tensor[0, 1, y, x] = c.Green / 255f;
                    tensor[0, 2, y, x] = c.Blue / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);

            // 解析 outputs：常见情形：一个输出为 [1,2] -> gender logits/probs；另一个为 [1] -> age
            float femaleProb = 0f;
            float age = 0f;

            foreach (var r in results)
            {
                var t = r.AsTensor<float>();
                var dims = t.Dimensions.ToArray();
                // gender logits/probs
                if (dims.Length == 2 && dims[0] == 1 && dims[1] == 2)
                {
                    var arr = t.ToArray();
                    var soft = Softmax(arr);
                    // 默认约定：index 1 = female 概率（若不对，请根据你的模型调整）
                    femaleProb = soft.Length > 1 ? soft[1] : soft[0];
                }
                else if (t.Length == 1)
                {
                    // age 直接输出标量
                    age = t.GetValue(0);
                }
                else if (dims.Length == 2 && dims[0] == 1 && dims[1] > 2)
                {
                    // 有些实现使用回归的多维输出，尝试取第一个元素作为 age
                    age = t.ToArray().FirstOrDefault();
                }
                // 其它情况可以按模型输出名判断：if (r.Name.Contains("age")) ...
            }

            return (femaleProb, age);
        }

        private static float[] Softmax(float[] logits)
        {
            var max = logits.Max();
            var exps = logits.Select(l => MathF.Exp(l - max)).ToArray();
            var sum = exps.Sum();
            if (sum == 0) return exps.Select(_ => 0f).ToArray();
            return exps.Select(e => e / sum).ToArray();
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}