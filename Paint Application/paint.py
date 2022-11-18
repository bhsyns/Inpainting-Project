from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk, Image, ImageDraw
import imageio as iio



class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):

        self.im=iio.imread('RGB.png')
        sizex , sizey = self.im.shape[:2]

        self.root = Tk()
        self.root.title('Paint')
        self.root.geometry('500x300')
        self.root.maxsize(100+sizex,sizey)
        self.root.minsize(100+sizex,sizey)
        
        
        self.paint_tools = Frame(self.root,width=100,height=sizey,relief=RIDGE,borderwidth=2)
        self.paint_tools.place(x=0,y=0)

        self.pen_logo = ImageTk.PhotoImage(Image.open('pen.png'))
        self.p = Label(self.paint_tools, text="pen",borderwidth=0,font=('verdana',10,'bold'))
        self.p.place(x=5,y=11)
        self.pen_button = Button(self.paint_tools,padx=6,image=self.pen_logo,borderwidth=2,command=self.use_pen)
        self.pen_button.place(x=60,y=10)


        self.eraser_logo = ImageTk.PhotoImage(Image.open('eraser.png'))
        self.e = Label(self.paint_tools, text='eraser',font=('verdana',10,'bold'))
        self.e.place(x=5,y=50)
        self.eraser_button = Button(self.paint_tools,image = self.eraser_logo,borderwidth=2,command=self.use_eraser)
        self.eraser_button.place(x=60,y=50)
        
        self.save_logo = ImageTk.PhotoImage(Image.open('save.png'))
        self.save = Label(self.paint_tools, text="Save",borderwidth=0,font=('verdana',10,'bold'))
        self.save.place(x=5,y=90)
        self.save_button = Button(self.paint_tools,padx=6,image=self.save_logo,borderwidth=2,command=self.save_as_png)
        self.save_button.place(x=60,y=90)


        self.pen_size = Label(self.paint_tools,text="Pen Size",font=('verdana',10,'bold'))
        self.pen_size.place(x=15,y=225)
        self.choose_size_button = Scale(self.paint_tools, from_=1, to=10, orient=VERTICAL)
        self.choose_size_button.place(x=20,y=125)

        self.lor = Canvas(self.root, bg='white', width=sizex, height=sizey,relief=RIDGE,borderwidth=0)
        self.lor.place(x=100,y=0)

        self.c = Canvas(self.root, bg='white', width=sizex, height=sizey,relief=RIDGE,borderwidth=0)
        self.c.place(x=100,y=0)
        bg = PhotoImage(file = "RGB.png")
        self.c .create_image( 0, 0, image = bg, anchor = "nw")

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)

            self.lor.create_line(self.old_x, self.old_y, event.x, event.y,
                    width=self.line_width, fill=paint_color,
                    capstyle=ROUND, smooth=TRUE, splinesteps=36)
            
            
            
        self.old_x = event.x
        self.old_y = event.y

    def save_as_png(self,fileName="mask"):
        # save postscipt image 
        self.lor.postscript(file = fileName + '.eps') 
        # use PIL to convert to PNG 
        img = Image.open(fileName + '.eps') 
        img.save(fileName + '.png', 'png') 

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()