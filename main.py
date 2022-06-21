import tkinter as tk
from tkinter import filedialog
import palmutil
import PIL
from PIL import Image, ImageTk, ImageOps
import cv2

class Application(tk.Frame):
    def __init__(self,master):
        super().__init__(master)
        self.pack()

        self.master.geometry("600x600")
        self.master.title("手相認識")
        self.canvas = tk.Canvas(self.master)
        self.canvas.pack(expand = True, fill = tk.BOTH)
        # self.canvas.bind('<Button-1>', self.canvas_click)
        self.img_file_name = None
        self.palm_org_img = None
        self.palm_result_img = None
        self.select_img_button = tk.Button(
            self.master,
            text='select image',
            command=self.select_img_file
        )
        self.select_img_button.pack(fill="x",padx=10,pady=10,side="left")
        self.start_palm_read_button = tk.Button(
            self.master,
            text='start',
            command=self.start_palm_read
        )
        self.start_palm_read_button.pack(fill="x",padx=10,pady=10,side="left")

    def select_img_file(self):
        typ = [('画像ファイル','*.png *.jpg')] 
        self.img_file_name = filedialog.askopenfilename(filetypes = typ)
        if self.img_file_name is not None:
            img = cv2.imread(self.img_file_name)
            if img is None:
                return
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            self.show_img(img)


    def palm_read(self)->bool:
        if self.img_file_name is None:
            return False
        self.palm_org_img,self.palm_result_img = palmutil.palm_read(self.img_file_name)
        if self.palm_org_img is None and self.palm_result_img is None:
            self.img_file_name = None
            return False
        return True

    def show_img(self,img):
        pil_image = PIL.Image.fromarray(img)
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        pil_image = ImageOps.pad(pil_image, (canvas_width, canvas_height))
        self.photo_image = PIL.ImageTk.PhotoImage(image=pil_image)
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,                   
            image=self.photo_image
        )
        return True

    def start_palm_read(self):
        if not self.palm_read():
            print("failed palm_read")
            return
        if self.palm_org_img is None and self.palm_result_img is None:
            return False
        self.palm_org_img = cv2.cvtColor(self.palm_org_img, cv2.COLOR_BGR2RGB)
        self.palm_result_img = cv2.cvtColor(self.palm_result_img, cv2.COLOR_BGR2RGB)
        img = cv2.hconcat((self.palm_org_img, self.palm_result_img))
        if not self.show_img(img):
            print("failed show img")
            return

def main():
    win = tk.Tk()
    app = Application(master=win)
    app.mainloop()


if __name__ == "__main__":
    main()