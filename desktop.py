import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import  os
import base64
from Extract_match import  EM_Sys
from desktop_bg_jpg import img as desktop_bg

class ImageApp:
    def __init__(self, master,bg_file_path):
        self.master = master
        master.title("特征匹配系统")

        bg_img = Image.open(bg_file_path)
        bg_img = bg_img.resize((1500,650))
        bg_img = ImageTk.PhotoImage(bg_img)
        self.em_sys=EM_Sys()
        self.img1_name=None
        self.img2_name=None
        self.img_pic1=None
        self.img_pic2=None
        self.exit=1
        ##########
        self.img1 = tk.Label(self.master)
        self.img1.place(x=0, y=0)
        self.img2 = tk.Label(self.master)
        self.img2.place(x=750, y=0)

        self.bg = tk.Label(master)
        self.bg.place(x=0, y=0)
        self.bg.config(image=bg_img)
        self.bg.image = bg_img
        # 创建两个按钮和一个标签
        self.button1 = tk.Button(master,width=20, height=2,bg='white' ,
                                 font=("Arial", 12),text="载入图片1",
                                 command=lambda:self.select_para(img_order='1'))
        self.button1.place(x=20,y=680)
        self.button2 = tk.Button(master,width=20, height=2, bg='white',
                                 font=("Arial", 12) ,text="载入图片2",
                                 command=lambda:self.select_para(img_order='2'))
        self.button2.place(x=20,y=740)

        self.button_show=tk.Button(master,width=30, height=3,bg='green',
                                   font=("Arial", 12) ,text='开始匹配',
                                   command=self.open_new_window)
        self.button_show.place(x=650,y=720)

        self.button_gaussi = tk.Button(master, width=24, height=2,
                                     font=("Arial", 10), text='显示图片1高斯金字塔(第1组)',
                                     command=self.show_gaussi)
        self.button_gaussi.place(x=300, y=730)


        self.button_exit=tk.Button(master,width=20, height=2,bg='red',
                                   font=("Arial", 12) ,text='退出',
                                   command=master.destroy)
        self.button_exit.place(x=1260,y=750)
    ########################## tool #########################
    def open_new_window(self):
        self.new_window = tk.Toplevel(self.master)
        self.new_window.geometry("300x200")
        self.new_window.title("参数选择窗口")
        label1 = tk.Label(self.new_window,text='特征点提取方法:')
        label1.place(x=10,y=0)
        ex_method=ttk.Combobox(self.new_window, values=("sift"))
        ex_method.set('sift')
        ex_method.place(x=130,y=0)

        label2 = tk.Label(self.new_window,text='特征点匹配方法:')
        label2.place(x=10,y=40)
        ma_method=ttk.Combobox(self.new_window, values=("BF",'FLANN'))
        ma_method.set('BF')
        ma_method.place(x=130,y=40)

        label3 = tk.Label(self.new_window,text="Lowe's ratio阈值:")
        label3.place(x=10,y=80)
        lowe_ra=tk.Entry(self.new_window)
        lowe_ra.insert(0,'0.75')
        lowe_ra.place(x=130,y=80)

        label4 = tk.Label(self.new_window,text="使用RANSAC优化?")
        label4.place(x=10,y=120)
        opts=tk.StringVar(self.new_window)
        opt1=tk.Radiobutton(self.new_window, text="YES", variable=opts, value=True)
        opt1.place(x=130,y=120)
        opt2=tk.Radiobutton(self.new_window, text="NO", variable=opts, value=False)
        opt2.place(x=220,y=120)
        opts.set(True)

        button=tk.Button(self.new_window,
                         text='确认',
                         command=lambda:self.extract_and_match(ex_method.get(),
                                                               ma_method.get(),
                                                               float(lowe_ra.get()),
                                                               opts.get()))
        button.place(x=150,y=150)

    def select_para(self,img_order='1'):
        self.para_window = tk.Toplevel(self.master)
        self.para_window.geometry("350x250")
        self.para_window.title("参数选择窗口")

        label1 = tk.Label(self.para_window,text='缩放比例(0.5-1.5):')
        label1.place(x=20,y=0)
        zoom_ra = tk.Entry(self.para_window)
        zoom_ra.insert(0, '1')
        zoom_ra.place(x=130, y=0)

        label2 = tk.Label(self.para_window,text='旋转角度(0-360):')
        label2.place(x=20,y=40)
        rot_ang = tk.Entry(self.para_window)
        rot_ang.insert(0, '0')
        rot_ang.place(x=130, y=40)

        label3 = tk.Label(self.para_window,text='x方向平移量:')
        label3.place(x=20,y=80)
        x_ = tk.Entry(self.para_window)
        x_.insert(0, '0')
        x_.place(x=130, y=80)

        label4 = tk.Label(self.para_window,text='y方向平移量:')
        label4.place(x=20,y=120)
        y_ = tk.Entry(self.para_window)
        y_.insert(0, '0')
        y_.place(x=130, y=120)

        label5 = tk.Label(self.para_window,text='光照(-255至255):')
        label5.place(x=20,y=160)
        illu = tk.Entry(self.para_window)
        illu.insert(0, '0')
        illu.place(x=130, y=160)

        button = tk.Button(self.para_window,
                            text='确认',
                            command=lambda: self.open_image(float(zoom_ra.get()),
                                                            int(rot_ang.get()),
                                                            int(x_.get()),
                                                            int(y_.get()),
                                                            int(illu.get()),
                                                            img_order=img_order))
        button.place(x=150, y=190)
    def open_image(self,scale,rotate,dx,dy,illu,img_order='1'):
        # 打开文件对话框，让用户选择一个图片文件
        self.para_window.destroy()
        if (scale<0.5) or (scale>1.5):
            messagebox.showwarning("警告", "缩放比例不符合要求！")
            return
        if (rotate<0) or (rotate>360):
            messagebox.showwarning("警告", "旋转角度不符合要求！")
            return
        if (illu<-255) or (illu>255):
            messagebox.showwarning("警告", "光照强度不符合要求！")
            return
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        img_name=file_path.split('/')[-1]
        img_name=img_name.split('.')[0][0:-1]
        # 读取并显示图片
        img_pic = Image.open(file_path)
        img_pic=img_pic.resize((int(700*scale), int(img_pic.size[1] * 700 * scale / img_pic.size[0])))
        img_pic = np.array(img_pic)
        M = cv2.getRotationMatrix2D((img_pic.shape[1] / 2, img_pic.shape[0] / 2), rotate, 1)
        M[0, 2] += dx
        M[1, 2] += dy
        img_pic = cv2.warpAffine(img_pic, M, (img_pic.shape[1], img_pic.shape[0]))
        img_pic = cv2.convertScaleAbs(img_pic, alpha=1, beta=illu)
        img_pic = Image.fromarray(img_pic)
        img = ImageTk.PhotoImage(img_pic)
        if self.exit:
            self.exit=0
            self.bg.destroy()
        if img_order == '1':
            self.button1.config(bg="lightblue")
            self.img1_name=img_name
            self.img_pic1=img_pic
            self.img1.config(image=img)
            self.img1.image = img  # 保持对图像对象的引用，否则图像将被垃圾回收器删除
        elif img_order =='2':
            self.button2.config(bg="lightblue")
            self.img2_name = img_name
            self.img_pic2 = img_pic
            self.img2.config(image=img)
            self.img2.image = img  # 保持对图像对象的引用，否则图像将被垃圾回收器删除
        self.em_sys.update_img(self.img_pic1, self.img_pic2)

    ########################## tool #########################
    def extract_and_match(self,ex_method,ma_method,lowe_ra,Use_RANSANC):
        self.new_window.destroy()
        if (self.img1_name==None) or (self.img2_name==None):
            messagebox.showwarning("警告", "输入图片为空！")
            return
        if self.img1_name!=self.img2_name:
            messagebox.showwarning("警告", "输入的两张图片名字不匹配！")
            return

        self.em_sys.extract(ex_method=ex_method)

        M,good_matches=self.em_sys.match(ma_method=ma_method,lowe_ra=lowe_ra,Use_RANSANC=Use_RANSANC)
        if M.size!=0:
            print('变换矩阵为:')
            print(M)
        else:
            print('?')
        if good_matches!=[]:
            self.em_sys.Draw_img_ans(good_matches,flag=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    def show_gaussi(self):
        if not self.img_pic1:
            messagebox.showwarning("警告", "请先读入图片一！")
            return
        self.em_sys.show_gaussi()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1500x800")
    tmp = open('bg.png', 'wb')  # 创建临时的文件
    tmp.write(base64.b64decode(desktop_bg))  ##把这个图片解码出来，写入文件中去。
    tmp.close()
    #bg_file_path='D:\Pycharm\Program\professionally_designed_sift\desktop_bg.jpg'
    bg_file_path = 'bg.png'
    app = ImageApp(root,bg_file_path)
    os.remove('bg.png')
    root.mainloop()