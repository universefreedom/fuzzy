using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO; // MemoryStream 사용을 위한 선언
namespace ho_pi
{
    public partial class Form1 : Form
    {
        //open.Filter = "Image Files(*.jpg; *.jpeg; *.png; *.bmp)|*.jpg; *.jpeg;*.png; *.bmp";
        //pictureBox1.Image = new Bitmap(open.FileName);

        public Form1()
        {
            InitializeComponent();
            // UI 초기화
            // 메인 Form 크기
            this.Size = new Size(590, 360);
            // PictureBox1 속성
            pictureBox1.Location = new Point(20, 40);
            pictureBox1.Size = new Size(456, 656);
            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox1.BorderStyle = BorderStyle.FixedSingle;
            // PictureBox2 속성
            pictureBox2.Location = new Point(496, 40);
            pictureBox2.Size = new Size(456, 656);
            pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox2.BorderStyle = BorderStyle.FixedSingle;
        }
        public byte[,] BitmapToByteArray2D(Bitmap bmp)
        {
            byte[,] bmpArray = new byte[bmp.Height * bmp.Width, 3];
            for (int x = 0; x < bmp.Width; x++)
                for (int y = 0; y < bmp.Height; y++)
                {
                    Color pixelColor = bmp.GetPixel(x, y);
                    bmpArray[y * bmp.Width + x, 0] = pixelColor.R;
                    bmpArray[y * bmp.Width + x, 1] = pixelColor.G;
                    bmpArray[y * bmp.Width + x, 2] = pixelColor.B;
                }
            return bmpArray;
        }
        public Bitmap byteArray2DToBitmap(byte[,] byteArray, int width, int height)
        {
            Bitmap newbmp = new Bitmap(width, height);
            for (int x = 0; x < width; x++)
                for (int y = 0; y < height; y++)
                {
                    Color newColor = Color.FromArgb(byteArray[y * width + x, 0],
                    byteArray[y * width + x, 1], byteArray[y * width + x, 2]);
                    newbmp.SetPixel(x, y, newColor);
                }
            return newbmp;
        }
        

        



        // S_recon = M ⊙ S_mem + (1 - M) ⊙ S_FAM
        private int[,] MaskedReconstruction(int[,] S_mem, double[,] S_FAM, int[,] M, int height, int width)
        {
            int[,] S_recon = new int[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    double val = M[i, j] * S_mem[i, j] + (1 - M[i, j]) * S_FAM[i, j];
                    S_recon[i, j] = Math.Min(255, Math.Max(0, (int)Math.Round(val)));
                }
            return S_recon;
        }


        // 유사도 계산: ∥~S∧S_i∥ / ∥~S∥
        private double ComputeSimilarity(int[,] S_noisy, int[,] S_i, int height, int width)
        {
            double minSum = 0;
            double noisyNorm = 0;

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int minVal = Math.Min(S_noisy[i, j], S_i[i, j]);
                    minSum += minVal * minVal;
                    noisyNorm += S_noisy[i, j] * S_noisy[i, j];
                }
            }
            return Math.Sqrt(minSum) / (Math.Sqrt(noisyNorm) + 1e-8);
        }

        // F(x) 계산
        private double[,] FAMRestore(int[,] S_noisy, List<int[,]> memoryPatterns, int height, int width)
        {
            double[,] result = new double[height, width];

            foreach (var S_i in memoryPatterns)
            {
                double sim = ComputeSimilarity(S_noisy, S_i, height, width);
                for (int i = 0; i < height; i++)
                    for (int j = 0; j < width; j++)
                        result[i, j] += sim * S_i[i, j];
            }
            return result;
        }
         
        

        List<double> Softmax(List<double> values)
        {
            double maxVal = values.Max(); // 수치 안정화용 최대값 뺄셈
            double sumExp = 0;
            List<double> expValues = new List<double>();

            foreach (var v in values)
            {
                double e = Math.Exp(v - maxVal);
                expValues.Add(e);
                sumExp += e;
            }

            List<double> softmax = expValues.Select(e => e / sumExp).ToList();
            return softmax;
        }

        double[,] FAMSoftmaxRestore(int[,] S_noisy, List<int[,]> memoryPatterns, int height, int width)
        {
            double[,] result = new double[height, width];

            // 1) 유사도 리스트 계산
            List<double> simList = new List<double>();
            foreach (var S_i in memoryPatterns)
                simList.Add(ComputeSimilarity(S_noisy, S_i, height, width));

            // 2) softmax 가중치 계산
            List<double> weights = Softmax(simList);

            // 3) 가중합으로 복원
            for (int k = 0; k < memoryPatterns.Count; k++)
            {
                var S_i = memoryPatterns[k];
                double w = weights[k];
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        result[i, j] += w * S_i[i, j];
                    }
                }
            }
            return result;
        }


        // Skip Connection 포함 최종 복원 S^ = αF(x) + (1-α)x
        private double[,] FinalOutput(int[,] S_noisy, List<int[,]> memoryPatterns, int height, int width, double alpha = 0.7)
        {   //FAMSoftmaxRestore//FAMRestore
            double[,] FAM = FAMSoftmaxRestore(S_noisy, memoryPatterns, height, width);
            double[,] result = new double[height, width];

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    result[i, j] = alpha * FAM[i, j] + (1 - alpha) * S_noisy[i, j];

            return result;
        }
        



        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
        void FuzzyStretch(ref Bitmap bitmap, int mode)
        {
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    Color c = bitmap.GetPixel(x, y);
                    int g = (c.R + c.G + c.B) / 3;
                    int stretched = g;

                    if (mode == 1 && g >= 180)
                        stretched = (g - 180) * 255 / (255 - 180);
                    else if (mode == -1 && g <= 80)
                        stretched = g * 255 / 80;

                    stretched = Math.Min(255, Math.Max(0, stretched));
                    bitmap.SetPixel(x, y, Color.FromArgb(stretched, stretched, stretched));
                }
            }
        }

        void PreprocessFuzzy(ref Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;

            double total = 0;
            int Xmin = 255, Xmax = 0;

            // 평균 밝기 및 최소/최대 계산
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    Color c = bitmap.GetPixel(x, y);
                    int g = (c.R + c.G + c.B) / 3;
                    total += g;
                    Xmin = Math.Min(Xmin, g);
                    Xmax = Math.Max(Xmax, g);
                }

            double I_tilde = total / (width * height);

            // 퍼지 스트레칭 (밝거나 어두운 이미지)
            if (I_tilde > 200)
                FuzzyStretch(ref bitmap, 1);
            else if (I_tilde < 60)
                FuzzyStretch(ref bitmap, -1);

            // 동적 알파컷 이진화 적용
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    Color color = bitmap.GetPixel(x, y);
                    int pixelGray = (color.R + color.G + color.B) / 3;

                    double l = 0.2 * I_tilde;
                    double m = 0.4 * Math.Abs(pixelGray - Xmin);
                    double o = 0.4 * Math.Abs(pixelGray - Xmax);
                    double alpha = Math.Min(l, Math.Min(m, o));
                    //255 : 0
                    int newVal = (pixelGray >= (I_tilde + alpha)) ? 170 : 80;
                    bitmap.SetPixel(x, y, Color.FromArgb(newVal, newVal, newVal));
                }
        }

        private void famToolStripMenuItem_Click_1(object sender, EventArgs e)
        {
            
            // 0. UI 설정
            pictureBox3.Location = new Point(1020, 40);
            pictureBox3.Size = new Size(456, 656);
            pictureBox3.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox3.BorderStyle = BorderStyle.FixedSingle;
            this.Size = new Size(1550, 720); // 화면 크기 조정

            // 1. 학습 이미지 선택 (Memory 저장용)
            OpenFileDialog open = new OpenFileDialog();
            open.Title = "학습할 이미지 선택";
            open.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
            if (open.ShowDialog() == DialogResult.OK)
            {
                pictureBox1.Image = new Bitmap(open.FileName);
            }

            // 2. 예측 이미지 선택 (손상된 이미지)
            open.Title = "예측할 이미지 선택";
            if (open.ShowDialog() == DialogResult.OK)
            {
                pictureBox2.Image = new Bitmap(open.FileName);
            }

            Bitmap imgLearn = (Bitmap)pictureBox1.Image;
            Bitmap imgNoisy = (Bitmap)pictureBox2.Image;

            // 2. 규격 통일 (가장 작은 쪽 기준으로 자르거나, 큰 쪽에 맞춰 리사이즈)
            int targetWidth = Math.Min(imgLearn.Width, imgNoisy.Width);
            int targetHeight = Math.Min(imgLearn.Height, imgNoisy.Height);

            // 3. 리사이징
            Bitmap resizedLearn = new Bitmap(imgLearn, new Size(targetWidth, targetHeight));
            Bitmap resizedNoisy = new Bitmap(imgNoisy, new Size(targetWidth, targetHeight));

            // 전처리 함수 호출
            PreprocessFuzzy(ref resizedNoisy);

            // 4. 배열 변환
            byte[,] learnArray = BitmapToByteArray2D(resizedLearn);
            byte[,] noisyArray = BitmapToByteArray2D(resizedNoisy);

            // 5. 그레이 패턴으로 복사
            int[,] learnPattern = new int[targetHeight, targetWidth];
            int[,] noisyPattern = new int[targetHeight, targetWidth];
            
            /*int height = imgLearn.Height;
            int width = imgLearn.Width;

            byte[,] learnArray = BitmapToByteArray2D(imgLearn);
            byte[,] noisyArray = BitmapToByteArray2D(imgNoisy);

            // 3. 이진화 또는 그레이화 (단순화)
            int[,] learnPattern = new int[height, width];
            int[,] noisyPattern = new int[height, width];*/

            
            for (int i = 0; i < targetHeight; i++)
                for (int j = 0; j < targetWidth; j++)
                {
                    learnPattern[i, j] = learnArray[i * targetWidth + j, 0];
                    noisyPattern[i, j] = noisyArray[i * targetWidth + j, 0];
                    
                }

            // 4. 메모리에 저장된 복수 패턴 생성 (예시: 하나만 저장)
            var memoryPatterns = new List<int[,]>() { learnPattern };

            /*List<double> simList = new List<double>();
            foreach (var S_i in memoryPatterns)
            {
                double sim = ComputeSimilarity(noisyPattern, S_i, height, width);
                simList.Add(sim);
            }*/
            // 5. FAM 기반 복원//FAMSoftmaxRestore//FAMRestore
            double[,] famOut = FAMSoftmaxRestore(noisyPattern, memoryPatterns, targetHeight, targetWidth);

            // 6. Skip Connection 포함 최종 출력
            double alpha = 0.7;
            double[,] S_hat = FinalOutput(noisyPattern, memoryPatterns, targetHeight, targetWidth, alpha);

            // 7. 마스크 생성 (손실된 영역 = 0, 나머지 = 1) → 단순 비교 기반
            int[,] M = new int[targetHeight, targetWidth];
            for (int i = 0; i < targetHeight; i++)
                for (int j = 0; j < targetWidth; j++)
                    M[i, j] = (noisyPattern[i, j] < 30) ? 0 : 1; // 30 이하는 손실로 간주

            // 8. 마스크 기반 최종 복원
            int[,] finalReconstruction = MaskedReconstruction(learnPattern, S_hat, M, targetHeight, targetWidth);

            // 9. 출력 이미지 생성
            byte[,] finalByte = new byte[targetHeight * targetWidth, 3];
            for (int i = 0; i < targetHeight; i++)
                for (int j = 0; j < targetWidth; j++)
                {
                    byte val = (byte)Math.Min(255, Math.Max(0, finalReconstruction[i, j]));
                    finalByte[i * targetWidth + j, 0] = val;
                    finalByte[i * targetWidth + j, 1] = val;
                    finalByte[i * targetWidth + j, 2] = val;
                }

            pictureBox3.Image = byteArray2DToBitmap(finalByte, targetWidth, targetHeight);


        }

        
        const int targetWidth = 128;
        const int targetHeight = 128;

        private Bitmap ResizeImage(Bitmap src, int width, int height)
        {
            Bitmap resized = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.DrawImage(src, 0, 0, width, height);
            }
            return resized;
        }
        // 1. NormalizeImageTo2DArray: 0~1로 정규화된 2D 배열 반환
        public double[,] NormalizeImageTo2DArray(Bitmap bmp)
        {
            int width = bmp.Width;
            int height = bmp.Height;
            double[,] array = new double[height, width];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    Color color = bmp.GetPixel(x, y);
                    array[y, x] = color.R / 255.0; // 그레이스케일 정규화
                }

            return array;
        }

        // 2. BuildFuzzyMemory2D: 퍼지 연상 메모리 (2D용)
        public double[,,] BuildFuzzyMemory2D(double[,] input, double[,] output, int height, int width)
        {
            double[,,] memory = new double[height, width, 2];

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    memory[i, j, 0] = input[i, j];  // 입력
                    memory[i, j, 1] = output[i, j]; // 출력
                }

            return memory;
        }

        // 3. TriangularMembership 그대로 사용
        private double TriangularMembership(double x, double center, double width = 0.3)
        {
            double a = center - width;
            double c = center + width;

            if (x <= a || x >= c) return 0;
            else if (x <= center) return (x - a) / (center - a);
            else return (c - x) / (c - center);
        }

        // 4. FuzzyRecall2D: 2D 기반 추론
        public double[,] FuzzyRecall2D(double[,] input, double[,,] memory, int height, int width)
        {
            double[,] output = new double[height, width];

            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                {
                    double x = input[i, j];
                    double center = memory[i, j, 0]; // 학습 입력
                    double y = memory[i, j, 1];      // 학습 출력

                    double alpha = TriangularMembership(x, center);
                    output[i, j] = alpha * y;
                }

            return output;
        }

        // 5. ToBitmap: 2D 배열 → 비트맵
        public Bitmap ToBitmap(double[,] array)
        {
            int height = array.GetLength(0);
            int width = array.GetLength(1);
            Bitmap bmp = new Bitmap(width, height);

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int val = (int)(array[y, x] * 255);
                    val = Math.Max(0, Math.Min(255, val));
                    Color color = Color.FromArgb(val, val, val);
                    bmp.SetPixel(x, y, color);
                }

            return bmp;
        }
        private double[,] ResidualConnection(double[,] input, double[,] recon, int height, int width, double alpha)
        {
            double[,] output = new double[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    output[i, j] = alpha * recon[i, j] + (1 - alpha) * input[i, j];
            return output;
        }
        private void 전통famToolStripMenuItem_Click(object sender, EventArgs e)
        {
            // 0. pictureBox3 설정 (UI에 반드시 보이도록)
            pictureBox3.Location = new Point(1020, 40);
            pictureBox3.Size = new Size(450, 650);  // 확대 가능
            pictureBox3.SizeMode = PictureBoxSizeMode.StretchImage;
            pictureBox3.BorderStyle = BorderStyle.FixedSingle;

            // 폼 크기도 넉넉하게 확보
            this.Size = new Size(1500, 750);  // 여유 있게 조정 (원래 1400x700)

            OpenFileDialog open = new OpenFileDialog();
            open.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";

            // 학습 이미지
            open.Title = "학습 이미지 선택";
            if (open.ShowDialog() != DialogResult.OK) return;
            Bitmap bmpLearn = ResizeImage(new Bitmap(open.FileName), targetWidth, targetHeight);
            //pictureBox1.Image = bmpLearn;
            
            // 정규화 (0~1 double 배열)
            double[,] learnArray = NormalizeImageTo2DArray(bmpLearn);

            // 잔차 연결 적용 (input과 output을 동일하게 넣음)
            double alpha = 0.7;
            double[,] learnResidual = ResidualConnection(learnArray, learnArray, targetHeight, targetWidth, alpha);

            // 다시 비트맵으로 변환해서 pictureBox1에 표시 (선명도 보존)
            pictureBox1.Image = ToBitmap(learnResidual);

            // 손상 이미지
            open.Title = "손상 이미지 선택";
            if (open.ShowDialog() != DialogResult.OK) return;
            Bitmap bmpNoisy = ResizeImage(new Bitmap(open.FileName), targetWidth, targetHeight);
            //pictureBox2.Image = bmpNoisy;
            
            // 정규화 (0~1 double 배열)
            double[,] learnArray1 = NormalizeImageTo2DArray(bmpNoisy);

            // 잔차 연결 적용 (input과 output을 동일하게 넣음)
            double[,] learnResidual1 = ResidualConnection(learnArray1, learnArray1, targetHeight, targetWidth, alpha);

            // 다시 비트맵으로 변환해서 pictureBox1에 표시 (선명도 보존)
            pictureBox2.Image = ToBitmap(learnResidual1);

            int width = targetWidth;
            int height = targetHeight;

            // 정규화
            double[,] input = NormalizeImageTo2DArray(bmpNoisy);
            double[,] output = NormalizeImageTo2DArray(bmpLearn);

            // 4. 퍼지 메모리 생성
            double[,,] famMemory = BuildFuzzyMemory2D(input, output, height, width);

            // 5. 전통 삼각 fam 추론
            double[,] famRecall = FuzzyRecall2D(input, famMemory, height, width);

            // 6. 잔차 연결 추가
            double beta = 0.7;
            double[,] newOutput = new double[height, width];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < width; j++)
                    newOutput[i, j] = beta * famRecall[i, j] + (1 - beta) * input[i, j];

            // 7. 출력 이미지 생성 및 표시
            Bitmap result = ToBitmap(newOutput);
            pictureBox3.Image = result;

            /*// 퍼지 메모리 생성
            double[,,] famMemory = BuildFuzzyMemory2D(input, output, height, width);

            // 추론
            double[,] recalled = FuzzyRecall2D(input, famMemory, height, width);

            // 이미지 출력
            Bitmap result = ToBitmap(recalled);
            pictureBox3.Image = result;*/


        }

    }

}
