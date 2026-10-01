namespace da
{
    partial class Form1
    {
        /// <summary>
        /// 필수 디자이너 변수입니다.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 사용 중인 모든 리소스를 정리합니다.
        /// </summary>
        /// <param name="disposing">관리되는 리소스를 삭제해야 하면 true이고, 그렇지 않으면 false입니다.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 디자이너에서 생성한 코드

        /// <summary>
        /// 디자이너 지원에 필요한 메서드입니다. 
        /// 이 메서드의 내용을 코드 편집기로 수정하지 마세요.
        /// </summary>
        private void InitializeComponent()
        {
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.메뉴ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.열기ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.저장ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripSeparator();
            this.종료ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.이진화ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.평균이진화ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.maxmin이진화ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.삼각형퍼지이진화ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.사다리꼴퍼지이진화ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.사다리꼴알파원본ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // menuStrip1
            // 
            this.menuStrip1.GripMargin = new System.Windows.Forms.Padding(2, 2, 0, 2);
            this.menuStrip1.ImageScalingSize = new System.Drawing.Size(24, 24);
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.메뉴ToolStripMenuItem,
            this.이진화ToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(1143, 35);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // 메뉴ToolStripMenuItem
            // 
            this.메뉴ToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.열기ToolStripMenuItem,
            this.저장ToolStripMenuItem,
            this.toolStripMenuItem1,
            this.종료ToolStripMenuItem});
            this.메뉴ToolStripMenuItem.Name = "메뉴ToolStripMenuItem";
            this.메뉴ToolStripMenuItem.Size = new System.Drawing.Size(64, 29);
            this.메뉴ToolStripMenuItem.Text = "메뉴";
            // 
            // 열기ToolStripMenuItem
            // 
            this.열기ToolStripMenuItem.Name = "열기ToolStripMenuItem";
            this.열기ToolStripMenuItem.Size = new System.Drawing.Size(150, 34);
            this.열기ToolStripMenuItem.Text = "열기";
            this.열기ToolStripMenuItem.Click += new System.EventHandler(this.열기ToolStripMenuItem_Click);
            // 
            // 저장ToolStripMenuItem
            // 
            this.저장ToolStripMenuItem.Name = "저장ToolStripMenuItem";
            this.저장ToolStripMenuItem.Size = new System.Drawing.Size(150, 34);
            this.저장ToolStripMenuItem.Text = "저장";
            this.저장ToolStripMenuItem.Click += new System.EventHandler(this.저장ToolStripMenuItem_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(147, 6);
            // 
            // 종료ToolStripMenuItem
            // 
            this.종료ToolStripMenuItem.Name = "종료ToolStripMenuItem";
            this.종료ToolStripMenuItem.Size = new System.Drawing.Size(150, 34);
            this.종료ToolStripMenuItem.Text = "종료";
            this.종료ToolStripMenuItem.Click += new System.EventHandler(this.종료ToolStripMenuItem_Click);
            // 
            // 이진화ToolStripMenuItem
            // 
            this.이진화ToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.평균이진화ToolStripMenuItem,
            this.maxmin이진화ToolStripMenuItem,
            this.삼각형퍼지이진화ToolStripMenuItem,
            this.사다리꼴퍼지이진화ToolStripMenuItem,
            this.사다리꼴알파원본ToolStripMenuItem});
            this.이진화ToolStripMenuItem.Name = "이진화ToolStripMenuItem";
            this.이진화ToolStripMenuItem.Size = new System.Drawing.Size(82, 29);
            this.이진화ToolStripMenuItem.Text = "이진화";
            // 
            // 평균이진화ToolStripMenuItem
            // 
            this.평균이진화ToolStripMenuItem.Name = "평균이진화ToolStripMenuItem";
            this.평균이진화ToolStripMenuItem.Size = new System.Drawing.Size(276, 34);
            this.평균이진화ToolStripMenuItem.Text = "평균이진화";
            this.평균이진화ToolStripMenuItem.Click += new System.EventHandler(this.평균이진화ToolStripMenuItem_Click);
            // 
            // maxmin이진화ToolStripMenuItem
            // 
            this.maxmin이진화ToolStripMenuItem.Name = "maxmin이진화ToolStripMenuItem";
            this.maxmin이진화ToolStripMenuItem.Size = new System.Drawing.Size(276, 34);
            this.maxmin이진화ToolStripMenuItem.Text = "max-min이진화";
            this.maxmin이진화ToolStripMenuItem.Click += new System.EventHandler(this.maxmin이진화ToolStripMenuItem_Click);
            // 
            // 삼각형퍼지이진화ToolStripMenuItem
            // 
            this.삼각형퍼지이진화ToolStripMenuItem.Name = "삼각형퍼지이진화ToolStripMenuItem";
            this.삼각형퍼지이진화ToolStripMenuItem.Size = new System.Drawing.Size(276, 34);
            this.삼각형퍼지이진화ToolStripMenuItem.Text = "삼각형퍼지이진화";
            this.삼각형퍼지이진화ToolStripMenuItem.Click += new System.EventHandler(this.삼각형퍼지이진화ToolStripMenuItem_Click);
            // 
            // 사다리꼴퍼지이진화ToolStripMenuItem
            // 
            this.사다리꼴퍼지이진화ToolStripMenuItem.Name = "사다리꼴퍼지이진화ToolStripMenuItem";
            this.사다리꼴퍼지이진화ToolStripMenuItem.Size = new System.Drawing.Size(276, 34);
            this.사다리꼴퍼지이진화ToolStripMenuItem.Text = "사다리꼴퍼지이진화";
            this.사다리꼴퍼지이진화ToolStripMenuItem.Click += new System.EventHandler(this.사다리꼴퍼지이진화ToolStripMenuItem_Click);
            // 
            // 사다리꼴알파원본ToolStripMenuItem
            // 
            this.사다리꼴알파원본ToolStripMenuItem.Name = "사다리꼴알파원본ToolStripMenuItem";
            this.사다리꼴알파원본ToolStripMenuItem.Size = new System.Drawing.Size(276, 34);
            this.사다리꼴알파원본ToolStripMenuItem.Text = "사다리꼴알파원본";
            this.사다리꼴알파원본ToolStripMenuItem.Click += new System.EventHandler(this.사다리꼴알파원본ToolStripMenuItem_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(10F, 18F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1143, 675);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Margin = new System.Windows.Forms.Padding(4, 4, 4, 4);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.Paint += new System.Windows.Forms.PaintEventHandler(this.Form1_Paint);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem 메뉴ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 열기ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 저장ToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem 종료ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 이진화ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 평균이진화ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem maxmin이진화ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 삼각형퍼지이진화ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 사다리꼴퍼지이진화ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 사다리꼴알파원본ToolStripMenuItem;
    }
}

