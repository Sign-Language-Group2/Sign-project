
import tkinter as tk
from PIL import ImageTk, Image
# from game_class import widgets

class RoundedButton(tk.Button):
    def __init__(self, *args, command=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config(relief=tk.FLAT)
        self.config(borderwidth=0)
        self.config(highlightthickness=0)
        self.config(background="#ff75c8")
        self.config(foreground="white")
        self.config(activebackground="#BF4698")
        self.config(activeforeground="white")
        self.config(cursor="hand2")
        self.config(font=("Comic Sans MS", 30))
        self.config(padx=25)
        self.config(pady=25)
        self.config(command=command)
        self.config(compound=tk.CENTER)

        # self.config(relief=tk.FLAT)
        # self.config(borderwidth=0)
        # self.config(highlightthickness=0)
        # self.config(background="#146C94")
        # self.config(foreground="white")
        # self.config(activebackground="#19A7CE")
        # self.config(activeforeground="white")
        # self.config(cursor="hand2")
        # self.config(font=("Comic Sans MS", 10))
        # self.config(padx=200)
        # self.config(pady=25)
        # self.config(command=self.on_click)
        # self.config(compound=tk.CENTER)
        # bg="#28393a",fg="white",cursor="hand2",activebackground="#146C94",background="#526D82",font=("Comic Sans MS", 10),padx=200,borderwidth=0,state=NORMAL
        
            
        self.bind("<Enter>", self.on_hover_enter)
        self.bind("<Leave>", self.on_hover_leave)
            
        self.bind("<Enter>", self.on_hover_enter)
        self.bind("<Leave>", self.on_hover_leave)
    
    # def on_click(self):
    #     if self == widgets["how_to_play_button"][0]:
    #         open_how_to_play()

    def on_hover_enter(self, event):
        self.config(font=("Comic Sans MS", 10))  # Increase font size to 38

    def on_hover_leave(self, event):
        self.config(background="#ff75c8")
        self.config(font=("Comic Sans MS", 10))  # Restore font size to 35

