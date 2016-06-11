using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BasicMnist
{
    public class DigitImage
    {
        public readonly byte[,] pixels;
        public readonly byte label;

        public DigitImage(byte[,] pixels, byte label)
        {
            this.pixels = (byte[,])pixels.Clone();
            this.label = label;
        }
    }

    public static class MnistParser
    {
        public const int ImageSize = 28;
        public const int LabelCount = 10; // each digit has one of 10 possible labels

        public static DigitImage[] Parse(string imageFile, string labelFile, int? maxImages)
        {
            var result = new List<DigitImage>();

            using (var brLabels = new BinaryReader(new FileStream(labelFile, FileMode.Open)))
            using (var brImages = new BinaryReader(new FileStream(imageFile, FileMode.Open)))
            {
                int magic1 = SwapEndian(brImages.ReadInt32());
                int numImages = SwapEndian(brImages.ReadInt32());
                int numRows = SwapEndian(brImages.ReadInt32());
                int numCols = SwapEndian(brImages.ReadInt32());

                int magic2 = SwapEndian(brLabels.ReadInt32());
                int numLabels = SwapEndian(brLabels.ReadInt32());

                var pixels = new byte[ImageSize, ImageSize];

                var images = maxImages.HasValue ? Math.Min(maxImages.Value, numImages) : numImages;

                for (int di = 0; di < images; ++di)
                {
                    for (int i = 0; i < ImageSize; ++i)
                    {
                        for (int j = 0; j < ImageSize; ++j)
                        {
                            byte b = brImages.ReadByte();
                            pixels[i, j] = b;
                        }
                    }

                    byte lbl = brLabels.ReadByte();

                    result.Add(new DigitImage(pixels, lbl));
                }
            }

            return result.ToArray();
        }

        private static int SwapEndian(int x)
        {
            return (int)SwapBytes((uint)x);
        }
        private static uint SwapBytes(uint x)
        {
            // swap adjacent 16-bit blocks
            x = (x >> 16) | (x << 16);
            // swap adjacent 8-bit blocks
            return ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
        }
    }
}
