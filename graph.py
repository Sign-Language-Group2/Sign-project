import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from Game.game_class import Game

# Custom frame with gradient background, buttons, and image
class GradientFrameWithButtonsAndImage(ttk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # Create a canvas for the gradient background
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        # Bind the canvas to the resize event
        self.canvas.bind('<Configure>', self.draw_gradient)

        # Create a new style for the buttons
        button_style = ttk.Style()
        button_style.configure('Custom.TButton', padding=10)  # Set padding for the buttons

        # Create two buttons with the custom style
        self.button1 = ttk.Button(self, text='Button 1', style='Custom.TButton', width=60)  # Maximize width by 6 times
        self.button2 = ttk.Button(self, text='Button 2', style='Custom.TButton', width=60)  # Maximize width by 6 times

        # Position the buttons near the bottom
        self.button1.place(relx=0.7, rely=0.9, anchor='center')
        self.button2.place(relx=0.2, rely=0.9, anchor='center')

        # Configure button2 to start the game
        self.button2.configure(command=self.start_game)

        # Load and display the image
        self.load_image()

    def load_image(self):
        # Load the image file
        image = Image.open('Sign-project\proud.jpg')  # Replace 'path_to_your_image.jpg' with the actual image file path

        # Calculate the size of the window
        window_width = self.winfo_width()
        window_height = self.winfo_height()

        # Calculate the new size for the image
        new_width = window_width * 600
        new_height = window_height * 230

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        # Convert the resized image to Tkinter-compatible format
        self.photo = ImageTk.PhotoImage(resized_image)

        # Create a label to display the image
        self.image_label = tk.Label(self, image=self.photo)
        self.image_label.place(relx=0.5, rely=0.5, anchor='center')

    def start_game(self):
        model_path = './Game/model/model.p'  # Replace with the actual path to your model file
        game = Game(model_path)
        game.start_game()

    def draw_gradient(self, event=None):
        # Retrieve the dimensions of the frame
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Define the gradient colors
        start_color = '#000000'  # Black
        end_color = '#8B00FF'  # Violet

        # Clear the canvas
        self.canvas.delete('gradient')

        # Draw the gradient rectangle
        for i in range(height):
            # Calculate the RGB values of the current row
            r = int((1 - i / height) * int(start_color[1:3], 16) + (i / height) * int(end_color[1:3], 16))
            g = int((1 - i / height) * int(start_color[3:5], 16) + (i / height) * int(end_color[3:5], 16))
            b = int((1 - i / height) * int(start_color[5:7], 16) + (i / height) * int(end_color[5:7], 16))

            # Convert the RGB values to hexadecimal
            color = f'#{r:02x}{g:02x}{b:02x}'

            # Draw a horizontal line with the gradient color
            self.canvas.create_line(0, i, width, i, fill=color, tags='gradient')

# Create the main window
root = tk.Tk()

# Create the gradient frame with buttons and image
gradient_frame = GradientFrameWithButtonsAndImage(root)
gradient_frame.pack(fill='both', expand=True)

# Start the Tkinter event loop
root.mainloop()
