using Microsoft.Win32;
using System.IO;
using System.Threading.Tasks;
using System.Windows;

namespace GinkgoBeautySelector
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow
    {
        public MainWindow()
        {
            InitializeComponent();


        }

        //private async void Button_Click(object sender, RoutedEventArgs e)
        //{
        //    // Create new Yolov8 predictor, specifying the model (in ONNX format)
        //    // If you are using a custom trained model, you can provide an array of labels. Otherwise, the standard Coco labels are used.
        //    using var yolo = YoloV8Predictor.Create("yolov8s.onnx");

        //    var folderDialog = new OpenFolderDialog();

        //    if (folderDialog.ShowDialog() == true)
        //    {
        //        var folderName = folderDialog.FolderName;
        //        var files = Directory.GetFiles(folderName, "*.jpg", SearchOption.AllDirectories);
        //        int i = 1;
        //        foreach (var file in files)
        //        {
        //            if (new FileInfo(file).Length > 100)
        //            {
        //                try
        //                {
        //                    using var image = Image.Load(file);
        //                    var predictions = yolo.Predict(image);

        //                    bool hasPerson = predictions.Any(p => p.Label.Name == "person" && p.Score > 0.5);
        //                    if (!hasPerson)
        //                        File.Delete(file);
        //                }
        //                catch { }
        //            }

        //            i++;
        //            await Task.Delay(100);
        //        }
        //    }
        //}

    }
}