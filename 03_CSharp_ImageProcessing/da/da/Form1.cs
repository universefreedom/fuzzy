using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace da
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            setShadowBitmap();
            copyBitmap2Array();
        }

        Graphics gr;
        Image image;
        Bitmap bitmap;
        const int HISTO_WIDTH = 256;
        const int HISTO_HEIGHT = 240;
        int[,] grayArray;
        int[,] ResultArray;
        //쉐도우 비트맵
        void setShadowBitmap()
        {
            bitmap = new Bitmap(ClientSize.Width, ClientSize.Height);
            gr = Graphics.FromImage(bitmap);
            gr.Clear(BackColor);
        }
        //카피
        void copyBitmap2Array()
        {
            Color color;
            int x, y, br;
            grayArray = new int[bitmap.Height, bitmap.Width];
            for (y = 0; y < bitmap.Height; y++)
                for (x = 0; x < bitmap.Width; x++)
                {
                    color = bitmap.GetPixel(x, y);
                    br = (int)(0.299 * color.R + 0.587 * color.G + 0.114 *
                    color.B);
                    grayArray[y, x] = br;
                }
        }
        //출력
        void displayArray(int leftTopX, int leftTopY, int[,] grayA)

        {
            int x, y;
            Color color;
            Bitmap gBitmap = new Bitmap(image.Width, image.Height);
            for (y = 0; y < image.Height; y++)
                for (x = 0; x < image.Width; x++)
                {
                    color = Color.FromArgb(grayA[y, x], grayA[y, x],
                    grayA[y, x]);
                    gBitmap.SetPixel(x, y, color);
                }
            gr.DrawImage(gBitmap, leftTopX, leftTopY, gBitmap.Width,
            gBitmap.Height);
            Invalidate();
        }
        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void 열기ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            gr = CreateGraphics();
            openFileDialog1.Title = "파일열기";
            openFileDialog1.Filter = "All File(*.*)|*.*|BitmapFile(*.bmp)|*.bmp";
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string str = openFileDialog1.FileName;
                image = Image.FromFile(str);
                setShadowBitmap();
                gr.DrawImage(image, 0, 0, image.Width, image.Height);
                copyBitmap2Array();
                // viewHistogram(image.Width + 10, 0, grayArray);
            }
            Invalidate();
        }

        private void 저장ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            saveFileDialog1.Title = "파일저장";
            saveFileDialog1.Filter = "All File(*.*)|*.*|BitmapFile(*.bmp)|*.bmp";
            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string str = saveFileDialog1.FileName;
                string strsaver = str.ToLower();
                image.Save(strsaver);
            }
        }

        private void 종료ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            Graphics grBm = e.Graphics;
            grBm.DrawImage(bitmap, 0, 0);
        }

        private void 평균이진화ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int x, y;
            int avg = 0;
            int count = 0;
            for (y = 0; y < bitmap.Height; y++)
                for (x = 0; x < bitmap.Width; x++)
                {
                    avg += grayArray[y, x];
                    count++;
                }
            avg = avg / count;
            for (y = 0; y < bitmap.Height; y++)
                for (x = 0; x < bitmap.Width; x++)
                {
                    if (grayArray[y, x] < avg)

                    {
                        grayArray[y, x] = 0;
                    }
                    else
                        grayArray[y, x] = 255;
                }
            displayArray(0, 0, grayArray);
        }

        private void maxmin이진화ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int x, y;
            int avg = 0;
            int max = 0, min = 256;
            for (y = 0; y < bitmap.Height; y++)
                for (x = 0; x < bitmap.Width; x++)
                {
                    if (max > grayArray[y, x])
                    {
                        grayArray[y, x] = max;
                    }
                    if (min < grayArray[y, x])
                    {
                        grayArray[y, x] = min;
                    }
                }
            avg = (max + min) / 2;
            for (y = 0; y < bitmap.Height; y++)
                for (x = 0; x < bitmap.Width; x++)
                {
                    if (grayArray[y, x] < avg)
                    {
                        grayArray[y, x] = 0;
                    }
                    else
                        grayArray[y, x] = 255;
                }
            displayArray(0, 0, grayArray);
        }

        private void 삼각형퍼지이진화ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int Xmean, Xmin = 255, Xmax = 0, Dmin, Dmax, a, Imax, Imin, Imid;
            //중간값, 최소값, 최대값, 밝은영역 거리값, 어두운 영역의 거리값, 조정률, 최대밝기값, 최소, 중간
            int x, y, r = 0, g = 0, b = 0;
            Color color;
            for (x = 0; x < image.Width; x++)
            {
                for (y = 0; y < image.Height; y++)
                {
                    color = bitmap.GetPixel(x, y);
                    r += color.R;
                    g += color.G;
                    b += color.B;
                    if (Xmin > (color.R + color.G + color.B) / 3)
                        Xmin = (color.R + color.G + color.B) / 3; //최소 RGB값 구하기
                    if (Xmax < (color.R + color.G + color.B) / 3)
                        Xmax = (color.R + color.G + color.B) / 3; //최대RGB값 구하기
                }
            }
            Xmean = ((r + g + b) / 3) / (image.Height * image.Width); //식(3) r * (1 / (image.Height * image.Width))
            Dmax = Math.Abs(Xmax - Xmean); //밝은 영역의 거리값 절대값
            Dmin = Math.Abs(Xmean - Xmin); //어두운 영역의 거리값
                                           //밝기의 조정률 구하기
            if (Xmin > 128) a = 255 - Xmean;
            else a = Xmin;
            if (Dmin > Xmean) a = Xmean;
            else a = Dmin;
            if (Dmax > Xmean) a = Xmean;
            else a = Dmax;
            Imax = Xmean + a; //밝기 조정률 a값으로 최대 밝기값
            Imin = Xmean - a; //최소 밝기값
            Imid = (Imax + Imin) / 2; //소속함수에서 소속도 1이 되기 위한 중간밝기값
            for (y = 0; y < image.Height; y++)
            {
                for (x = 0; x < image.Width; x++)
                {
                    if (fuzzy(Imin, Imax, Imid, grayArray[y, x]) > 0.5) //소속함수에서 구해진 소속도를 적용하여 영상에 적용
                    {
                        grayArray[y, x] = 0;
                    }
                    else
                    {
                        grayArray[y, x] = 255;
                    }
                }
            }
            displayArray(0, 0, grayArray);
        }
        double fuzzy(double Imin, double Imax, double Imid, double p)//소속도에 결정
        {
            double result = 0;
            if (p <= Imin || p >= Imax)
            {
                result = 0;
            }
            else if (p > Imid)
            {
                result = (Imax - p) / (Imax - Imid);
            }
            else if (p < Imid)
            {
                result = (p - Imin) / (Imid - Imin);
            }
            else if (p == Imid)
            {
                result = 1;
            }
            return result;
        }

        private int CountBinaryPixels(int[,] grayArray, int width, int height)
        {
            int count = 0;

            // 흰색 픽셀(255)의 수를 세기
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (grayArray[y, x] == 0) // 검은색 픽셀은 0으로 가정
                    {
                        count++;
                    }
                }
            }

            return count;
        }

        private void 사다리꼴퍼지이진화ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int M = image.Width; // 이미지 너비
            int N = image.Height; // 이미지 높이
            int n = 1; // 단계 인덱스 초기화
            int S_n = 1; // 초기 분할 개수
            int B_n = 1; // 총 블록 수
            double P_bin; // 이진화된 화소 수 비율

            int[,] grayArray = new int[N, M]; // 그레이스케일 배열 초기화

            // 색상 정보 수집 및 초기 값 설정
            int Xmin = 255, Xmax = 0, r = 0, g = 0, b = 0;
            Color color;

            for (int x = 0; x < M; x++)
            {
                for (int y = 0; y < N; y++)
                {
                    color = bitmap.GetPixel(x, y);
                    r += color.R;
                    g += color.G;
                    b += color.B;
                    int averageColor = (color.R + color.G + color.B) / 3;

                    if (Xmin > averageColor) Xmin = averageColor; // 최소 RGB값
                    if (Xmax < averageColor) Xmax = averageColor; // 최대 RGB값
                }
            }

            // 평균 값 계산
            double I_tilde = (double)(r + g + b) / (3 * M * N); // I_~

            // 블록 퍼지 이진화 수행
            while (true)
            {
                // S_n 계산
                if (n == 1)
                {
                    S_n = 1; // n=1일 경우, 블록 하나.
                }
                else
                {
                    S_n = 다음소수(n); // n번째 소수 (예: 2, 3, 5 등)
                }

                B_n = (int)Math.Pow(S_n, 2); // B_n = S_n^2

                // 각 블록에 대해 이진화 수행
                int W_n = M / S_n; // 블록 너비
                int H_n = N / S_n; // 블록 높이

                for (int a = 0; a < B_n; a++)
                {
                    int blockX = (a % S_n) * W_n; // 블록 시작 X 좌표
                    int blockY = (a / S_n) * H_n; // 블록 시작 Y 좌표

                    // 블록 내의 이진화 수행
                    for (int y = blockY; y < blockY + H_n; y++)
                    {
                        for (int x = blockX; x < blockX + W_n; x++)
                        {
                            color = bitmap.GetPixel(x, y);
                            int pixelGray = (color.R + color.G + color.B) / 3; // 현재 픽셀의 그레이값
                            double l = 0.2*I_tilde;
                            double m = 0.4 * Math.Abs(pixelGray - Xmin);
                            double o = 0.4 * Math.Abs(pixelGray - Xmax);
                            double α;
                            if (l < m && l < o)
                            {
                                α = l;
                            }
                            else if (m < o)
                            {
                                α = m;
                            }
                            else
                            {
                                α = o;
                            }

                            // 조정된 밝기 기준 계산 (동적 알파컷 적용)
                            if (pixelGray >= (I_tilde + α))
                            {
                                grayArray[y, x] = 255; // 흰색
                            }
                            else
                            {
                                grayArray[y, x] = 0; // 검정색
                            }
                        }
                    }
                }

                // P_bin 계산
                int binCount = CountBinaryPixels(grayArray, W_n, H_n);
                P_bin = (double)binCount / (W_n * H_n) * 100;

                // P_bin 조건에 따른 블록 생성
                if (P_bin <= 98)
                {
                    break; // 블록 생성
                }

                n++; // 단계 증가
            }

            displayArray(0, 0, grayArray); // 이진화한 이미지를 화면에 출력
        }

        private int 다음소수(int n)
        {
            // 소수 확인 및 반환하는 메서드
            int value = n + 1;
            while (true)
            {
                bool isPrime = true;
                for (int i = 2; i <= Math.Sqrt(value); i++)
                {
                    if (value % i == 0)
                    {
                        isPrime = false;
                        break;
                    }
                }

                if (isPrime)
                    return value;

                value++;
            }
        }


        double fuzzy1(double Imin, double Imax, double Immin, double Immin2, double Imid, double p)//소속도에 결정
        {
            //(double Imin,double Imax, double Imm, double Imx, double p
            double result = 0;
            if (p <= Imin || p >= Imax)
            {
                result = 0;
            }
            else if (p > Immin2 && p < Imax)
            {

                result = (Imax - p) / (Imax - Imid);
            }
            else if (p < Immin && p > Imin)
            {
                result = (p - Imin) / (Imid - Imin);
            }
            else if (p >= Immin && p <= Immin2)
            {
                result = 1;
            }
            return result;
        }

        private void 사다리꼴알파원본ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            int Xmin = 255, Xmax = 0, Dmin, Dmax, a, Imax, Imin, Immin, Immin2, m1, m2, w1, w2, Imid;
            double a1, a2;
            //중간값, 최소값, 최대값, 밝은영역 거리값, 어두운 영역의 거리값, 조정률, 최대밝기값, 최소, 중간
            int x, y, r = 0, g = 0, b = 0;
            Color color;
            for (x = 0; x < image.Width; x++)
            {
                for (y = 0; y < image.Height; y++)
                {
                    color = bitmap.GetPixel(x, y);
                    r += color.R;
                    g += color.G;
                    b += color.B;
                    if (Xmin > (color.R + color.G + color.B) / 3)
                        Xmin = (color.R + color.G + color.B) / 3; //최소 RGB값 구하기
                    if (Xmax < (color.R + color.G + color.B) / 3)
                        Xmax = (color.R + color.G + color.B) / 3; //최대RGB값 구하기
                }
            }
            m1 = (Xmin + Xmax) / 2;
            m2 = ((r + g + b) / 3) / (image.Height * image.Width); //식(3) r*(1 / (image.Height * image.Width))->M2//전체 이미지 평균 값// (/3) -> 컬러 값이 라서
            a1 = (m1 + m2) / 2;
            a2 = (a1 / 255);
            Dmax = Math.Abs(Xmax - m2); //밝은 영역의 거리값 절대값
            Dmin = Math.Abs(m2 - Xmin); //어두운 영역의 거리값
            w1 = Math.Abs(m1 - m2);
            w2 = Math.Abs(m1 + m2);
            //밝기의 조정률 적용
            if (Xmin > 128) a = 255 - m2;
            else a = Xmin;
            if (Dmin > m2) a = m2;
            else a = Dmin;


            if (Dmax > m2) a = m2;
            else a = Dmax;
            Imax = m2 + a; //밝기 조정률 a값으로 최대 밝기값
            Imin = m2 - a; //최소 밝기값
            Imid = (Imax + Imin) / 2; //소속함수에서 소속도 1이 되기 위한 중간밝기값
            Immin = Imid - w1;
            Immin2 = Imid - w2;


            for (y = 0; y < image.Height; y++)
            {
                for (x = 0; x < image.Width; x++)
                {
                    if (fuzzy1(Imin, Imax, Immin, Immin2, Imid, grayArray[y, x]) >
                    (0.5 + a2))//소속함수에서 구해진 소속도를 적용하여 영상에 적용
                               // 0.5 는 a 값
                               // (imin,Imax,Imm,Imx,grayArray[y,x]
                    {
                        grayArray[y, x] = 0;
                    }
                    else
                    {
                        grayArray[y, x] = 255;
                    }
                }
            }
            displayArray(0, 0, grayArray);
        
        }
    }
    
}
