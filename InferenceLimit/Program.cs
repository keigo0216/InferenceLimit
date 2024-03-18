// See https://aka.ms/new-console-template for more information
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Numerics.Tensors;
using System.Text.Json.Serialization;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

class Program
{
    static string modelPath = "classification_fgvc_model.onnx";
    static string imagePath = "imgs\\20220525162728376153.jpg";
    static InferenceSession session = new InferenceSession(modelPath);

    static void Main(string[] args)
    {
        var bitmap = new Bitmap(imagePath);
        var tensor = Tool.BitmapToTensor(bitmap);
        var count = 0;

        var tasks = new List<Task>();
        using (Profiler.BeginEvent("Inference by multi threading"))
        {
            while (count < 10000)
            {
                for (int i = 0; i < 8; i++)
                {
                    int threadId = i;
                    tasks.Add(Task.Run(() => RunInference(threadId, tensor)));
                }
                Task.WaitAll(tasks.ToArray());
                count++;
            }

        }

        session.Dispose();
    }

    private static void RunInference(int threasdId, DenseTensor<byte> tensor)
    {
        var count = 0;
        while (count < 1000)
        {
            int[] tensorDimensions = tensor.Dimensions.ToArray();
            long[] shape = Array.ConvertAll(tensorDimensions, dim => (long)dim);

            var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.Buffer, shape);

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_img", inputOrtValue }
            };

            var runOptions = new RunOptions();


            
            using (Profiler.BeginEvent($"Inference in threadid: {threasdId}"))
            {
                var output = session.Run(runOptions, inputs, session.OutputNames);
                var output_0 = output[0];
                var outputData = output_0.GetTensorDataAsSpan<float>();
            };

            count++;

            Console.WriteLine($"ThreadID: {threasdId}");
        } 
    }
}

class Tool
{
    public static DenseTensor<byte> BitmapToTensor(Bitmap image)
    {
        var rect = new Rectangle(0, 0, image.Width, image.Height);
        var bmpData = image.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

        try
        {
            var height = bmpData.Height;
            var width = bmpData.Width;

            // RGB画像のため、1ピクセルあたり3バイトを想定
            var rawArray = new byte[width * height * 3];
            Marshal.Copy(bmpData.Scan0, rawArray, 0, rawArray.Length);

            // テンソルの形状を[高さ, 幅, チャネル]に変更
            var tensor = new DenseTensor<byte>(new[] { height, width, 3 });
            int stride = bmpData.Stride;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * stride + x * 3;
                    tensor[y, x, 0] = rawArray[idx + 2]; // Rチャネル
                    tensor[y, x, 1] = rawArray[idx + 1]; // Gチャネル
                    tensor[y, x, 2] = rawArray[idx];     // Bチャネル
                }
            }
            return tensor;
        }
        finally
        {
            image.UnlockBits(bmpData);
        }
    }
}
