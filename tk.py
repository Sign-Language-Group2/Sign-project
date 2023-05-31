import sys
sys.path.insert(1, r'c:\users\myips\desktop\project\.venv\lib\site-packages')
from tkinter import *
from ttkbootstrap.constants import *
import ttkbootstrap as tb
from PIL import ImageTk , Image
import customtkinter 

#  create root window
root = tb.Window(themename='vapor')
root.title('SignSaga')
root.iconbitmap(r'./game_img/logo.ico')

# create main frame
bg_img=ImageTk.PhotoImage(Image.open(r'C:\Users\myips\Desktop\project\game_img\357ce415-aa49-4bb8-80a5-732c1bf1311b.jpg'))

main_frame=tb.Frame(root  )  # main_frame
main_frame.pack(padx=0,pady=0,fill=BOTH, expand=True)

# main_=tb.Label(main_frame,image=bg_img).place(x=0,y=0 ,relwidth=1,relheight=1)
# main_frame.configure()
# main_frame.config(width=main_frame.winfo_width(),height=main_frame.winfo_height())  #(width=background_img.width(), height=background_img.height())
# background_label =tb.Label(main_frame, image=bg_img , bootstyle="inverse")
# background_label.pack(padx=0,pady=0,fill=BOTH, expand=True)
#  style for button
btn_style=tb.Style()
btn_style.configure('primary.Outline.TButton',font=('Helvetica',20))
# create button in main frame
how_to_ply_btn=tb.Button(main_frame,text='how to play' ,bootstyle='outline' , style='primary.Outline.TButton',command='')     
how_to_ply_btn.pack(pady=20)
start_game_btn=tb.Button(main_frame,text='start game',  bootstyle='primary ,outline', command='')
start_game_btn.pack(pady=20)



root.mainloop()