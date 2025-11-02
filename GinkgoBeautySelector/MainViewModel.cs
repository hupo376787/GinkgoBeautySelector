using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;
using SkiaSharp;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using YoloDotNet;
using YoloDotNet.Core;
using YoloDotNet.Enums;
using YoloDotNet.Models;
using YoloDotNet.Models.Interfaces;

namespace GinkgoBeautySelector
{
    public partial class MainViewModel : ObservableObject
    {
        [ObservableProperty]
        private bool isUsingGPU = false;
        [ObservableProperty]
        private bool isIncludeSubFolder = true;
        [ObservableProperty]
        private string selectedFolderDescription;
        [ObservableProperty]
        private bool isDeleteNoneHuman = true;
        [ObservableProperty]
        private bool isOnlyKeepFemale;
        [ObservableProperty]
        private string statusDescription;
        [ObservableProperty]
        private double progress;

        Yolo yolo;
        GenderAgePredictor? genderAge;

        public MainViewModel()
        {
            InitYolo();
            InitGenderAge();
        }

        private void InitYolo()
        {
            IExecutionProvider provider = IsUsingGPU
                ? new CudaExecutionProvider(GpuId: 0, PrimeGpu: true)
                : new CpuExecutionProvider();

            yolo = new Yolo(new YoloOptions
            {
                OnnxModel = "yolo11m.onnx",
                ExecutionProvider = provider,
                ImageResize = ImageResize.Proportional,
                SamplingOptions = new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.None)
            });
            Debug.WriteLine($"Model Type: {yolo.ModelInfo}; EP={(IsUsingGPU ? "CUDA" : "CPU")}");
        }

        private void InitGenderAge()
        {
            var path = "genderage.onnx"; // 请确保模型文件放在可访问路径
            if (File.Exists(path))
            {
                try
                {
                    // 如果希望使用 GPU，请把第 2 个参数改为 true 并安装 OnnxRuntime.Gpu 包
                    genderAge = new GenderAgePredictor(path, useCuda: false, inputW: 112, inputH: 112);
                    Debug.WriteLine("Loaded genderage.onnx");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine("加载 genderage.onnx 失败：" + ex);
                    genderAge = null;
                }
            }
            else
            {
                Debug.WriteLine("未找到 genderage.onnx");
            }
        }

        [RelayCommand]
        private void SelectFolder()
        {
            var folderDialog = new OpenFolderDialog();

            if (folderDialog.ShowDialog() == true)
            {
                SelectedFolderDescription = folderDialog.FolderName;
            }
        }

        [RelayCommand]
        private async Task Run()
        {
            if (string.IsNullOrEmpty(SelectedFolderDescription))
                return;

            var patterns = new[] { "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp", "*.wbmp", "*.heif", "*.dng", "*.ktx", "*.pkm" };
            var searchOption = IsIncludeSubFolder ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

            var files = patterns
                .SelectMany(p => Directory.GetFiles(SelectedFolderDescription, p, searchOption))
                .Distinct(System.StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (files.Length == 0)
            {
                StatusDescription = "未找到图片文件。";
                Progress = 0;
                return;
            }

            int deletedCount = 0;
            for (int idx = 0; idx < files.Length; idx++)
            {
                var file = files[idx];
                if (File.Exists(file))
                {
                    try
                    {
                        using var image = SKBitmap.Decode(file);

                        // Run object detection
                        var results = yolo.RunObjectDetection(image, confidence: 0.50, iou: 0.7);

                        // 判断是否包含人类目标（类名 "person"）
                        bool hasPerson = ResultsContainPerson(results, minScore: 0.50);
                        Debug.WriteLine($"{file}, hasPerson={hasPerson}");

                        if (!hasPerson && IsDeleteNoneHuman)
                        {
                            try
                            {
                                File.Delete(file);
                                deletedCount++;
                            }
                            catch (System.Exception ex)
                            {
                                Debug.WriteLine($"删除文件失败：{file}，原因：{ex.Message}");
                            }
                        }
                    }
                    catch (System.Exception ex)
                    {
                        Debug.WriteLine($"处理图片失败：{file}，原因：{ex.Message}");
                    }
                }

                // 更新进度与状态
                Progress = (idx + 1) * 1.0 / files.Length;
                StatusDescription = $"处理 {idx + 1}/{files.Length}：{Path.GetFileName(file)}";
                await Task.Delay(100);
            }

            StatusDescription = $"总文件数：{files.Length}，已删除：{deletedCount}";
        }

        /// <summary>
        /// 通过反射检查检测结果集合中是否包含 label 为 "person" 的项。
        /// 兼容不同结果类型（Label 字符串 / Label 对象具 Name 属性等），并尝试读取 Score/Confidence 字段用于阈值比较。
        /// </summary>
        private static bool ResultsContainPerson(object runResults, double minScore = 0.5)
        {
            if (runResults is System.Collections.IEnumerable enumerable)
            {
                foreach (var item in enumerable)
                {
                    if (item == null) continue;

                    var t = item.GetType();

                    // 尝试获取 label 字段/属性
                    string? labelText = null;
                    PropertyInfo? labelProp = t.GetProperty("Label") ??
                                              t.GetProperty("LabelName") ??
                                              t.GetProperty("Name") ??
                                              t.GetProperty("ClassName") ??
                                              t.GetProperty("Category");
                    if (labelProp != null)
                    {
                        var val = labelProp.GetValue(item);
                        if (val != null)
                        {
                            if (val is string s) labelText = s;
                            else
                            {
                                // 若 Label 本身是对象，尝试读取其 Name 属性
                                var nameProp = val.GetType().GetProperty("Name");
                                if (nameProp != null)
                                    labelText = nameProp.GetValue(val)?.ToString();
                            }
                        }
                    }

                    // 尝试获取 score/confidence
                    double score = 1.0;
                    PropertyInfo? scoreProp = t.GetProperty("Score") ??
                                               t.GetProperty("Confidence") ??
                                               t.GetProperty("Prob") ??
                                               t.GetProperty("Probability");
                    if (scoreProp != null)
                    {
                        var sval = scoreProp.GetValue(item);
                        if (sval != null && double.TryParse(sval.ToString(), out var d)) score = d;
                    }

                    if (!string.IsNullOrEmpty(labelText) &&
                        labelText.Equals("person", System.StringComparison.OrdinalIgnoreCase) &&
                        score >= minScore)
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }
}
