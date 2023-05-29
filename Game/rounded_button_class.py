
import tkinter as tk
from PIL import ImageTk, Image


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
            
        self.bind("<Enter>", self.on_hover_enter)
        self.bind("<Leave>", self.on_hover_leave)
            
        self.bind("<Enter>", self.on_hover_enter)
        self.bind("<Leave>", self.on_hover_leave)

    def on_hover_enter(self, event):
        self.config(font=("Comic Sans MS", 33))  # Increase font size to 38

    def on_hover_leave(self, event):
        self.config(background="#ff75c8")
        self.config(font=("Comic Sans MS", 30))  # Restore font size to 35

