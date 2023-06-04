import pickle
import random
import cv2
import mediapipe as mp
import numpy as np
import time
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import threading
from Game.user_CSVManager_class import UserCSVManager
import pygame


import customtkinter

# Initialize Pygame mixer
pygame.mixer.init()


class Game:
    def __init__(self, model_path):
        # Load the model from the specified path using pickle
        self.model = pickle.load(open(model_path, 'rb'))['model']

        # Mapping of integer labels to corresponding characters
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
            5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }

        # Initialize MediaPipe hands, drawing, and drawing styles
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Open the video capture object for the default camera (index 0)
        self.cap = cv2.VideoCapture(0)

        # Store a randomly chosen character
        self.random_character = None

        # Store the current predicted character
        self.current_prediction_character = None

        # Temporary storage for a predicted character
        self.temp_character = None

        # Counter to track the number of times a predicted character matches the temporary character
        self.counter = 0
        self.two_hand = None
        self.total_game_time_seconds = 0
        self.random_character_change_time_seconds = 0
        self.total_score = 0

        self.Max_score_level_1 = 0
        self.Max_score_level_2 = 0
        self.Max_score_level_3 = 0

        self.game_terminating = False

        self.image_tk = None 
        self.opening_image_tk= None

        self.game_menu_window = None
        self.Main_window = None
        self.learn_level_window = None
        self.random_level_window = None
        self.score_window = None 

        self.canvas= None
        self.camera_thread= None

        self.learn_button = None
        self.Start_Play_button = None
        self.open_score_button = None

        self.random_1_button = None
        self.random_2_button = None
        self.random_3_button = None

        self.back_button = None
        self.back_button_2 = None
        self.default_sound_value = 10

        self.background_color="#212121"
        self.text_color="#59e7ed"
        self.border_color='#323232'
        self.logo_path="./Game/game_data/logo-3.png"
        self.stopwatch_path ="./Game/game_data/stopwatch.png"
        self.background_image_path ="./Game/game_data/BackgroundImage.png"
        self.sound_path ="./Game/game_data/sound.mp3"
        self.user_data_path = "./Game/game_data/users.csv"

        self.user_name = None
        self.csv_manager = None
        self.user_dropdown = None
        
    def generate_random_character(self):
        # Randomly choose a character from the labels_dict values
        self.random_character = random.choice(list(self.labels_dict.values()))

    def read_frame(self):
        # Read a frame from the video capture object
        ret, frame = self.cap.read()
        return frame
    
    def update_volume(self,value):
        volume = float(value) / 100
        pygame.mixer.music.set_volume(volume)

    def update_current_prediction_character(self, prediction):
        # Get the predicted character based on the prediction
        predicted_character = self.labels_dict[prediction]

        # If the temporary character is None or different from the predicted character, reset the counter and update the temporary character
        if self.temp_character is None or self.temp_character != predicted_character:
            self.temp_character = predicted_character
            self.counter = 1
        else:
            self.counter += 1

        # If the counter reaches 6, update the current prediction character and reset the temporary character
        if self.counter == 6:
            self.current_prediction_character = self.temp_character
            self.temp_character = None
            self.counter = 0
          
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        # Draw hand landmarks on the frame
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style())

    def extract_hand_data(self, hand_landmarks):
        # Extract relevant data from hand landmarks
        data_aux = []
        x_ = []
        y_ = []

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        return data_aux, x_, y_

    def detect_gesture(self, frame):
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) == 1:
                    self.two_hand=False
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks on the frame
                        self.draw_hand_landmarks(frame, hand_landmarks)
                    for hand_landmarks in results.multi_hand_landmarks:
                        data_aux, x_, y_ = self.extract_hand_data(hand_landmarks)
                        self.two_hand=False
                        x1 = int(min(x_) * W) - 10
                        y1 = int(min(y_) * H) - 10

                        x2 = int(max(x_) * W) - 10
                        y2 = int(max(y_) * H) - 10

                        # Predict the gesture
                        prediction = self.model.predict([np.asarray(data_aux)])
                        predicted_character = self.labels_dict[int(prediction[0])]

                        # update current prediction character
                        self.update_current_prediction_character(prediction[0])

                        # Draw a rectangle around the hand and display the predicted character
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                        
                else:
                    # Two hand landmarks
                    self.two_hand=True
                    self.temp_character=None
                    self.counter=None
                    self.current_prediction_character=None

            else:
                # No hand landmarks
                self.temp_character=None
                self.counter=None
                self.current_prediction_character=None


    def calculate_score(self,prediction_time_seconds):
        if prediction_time_seconds <= 2:
            return 3
        else:
            return 1
        
    def time_left_format(self,seconds):
        if seconds is not None:
            seconds = int(seconds)
            if(seconds<=0):
                return '-'
            d = seconds // (3600 * 24)
            h = seconds // 3600 % 24
            m = seconds % 3600 // 60
            s = seconds % 3600 % 60
            if d > 0:
                return '{:02d}:{:02d}:{:02d}:{:02d}'.format(d, h, m, s)
            elif h > 0:
                return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
            elif m > 0:
                return '{:02d}:{:02d}'.format(m, s)
            elif s > 0:
                return '{:02d}:{:02d}'.format(0,s)
        return '-'


    def random_level(self, total_time=15, character_change_time=5,game_level=1):

        self.game_terminating = False

        def on_closing():
            self.game_terminating = True

                    
        def update_camera():
            # disabled buttons:
            self.random_1_button.configure(state='disabled')
            self.random_2_button.configure(state='disabled')
            self.random_3_button.configure(state='disabled')
            self.back_button.configure(state='disabled')

            # camera fix
            ret, frame = self.cap.read()
            if not ret:
                self.cap = cv2.VideoCapture(0)

            # game logic
            # --------------------------------
             # Set total game time and character change time
            self.total_game_time_seconds = total_time
            self.random_character_change_time_seconds = character_change_time

            # Start the game timer
            game_start_time = time.time()
            prediction_start_time = time.time()
            prediction_time_seconds = 0

            # Generate initial random character
            self.generate_random_character()

            # Set initial values for the components
            if game_level==1:
                max_score_value.config(text = str(self.Max_score_level_1))
            elif game_level==2:
                max_score_value.config(text = str(self.Max_score_level_2))
            elif game_level==3:
                max_score_value.config(text = str(self.Max_score_level_3))

            character_value.config(text = self.random_character)
            total_score_value.config(text = str(self.total_score))


            # Calculate the angle based on the remaining time
            total_angle = 360 - (360 / self.total_game_time_seconds) * (self.total_game_time_seconds - 0)
            total_stop_watch_canvas.itemconfig(total_progress_arc, extent=-total_angle)
            total_time_value.config(text=self.time_left_format(self.total_game_time_seconds))  # Zero-padding for single digit seconds

            # Calculate the angle based on the remaining time 
            prediction_angle = 360 - (360 / self.random_character_change_time_seconds) * (self.random_character_change_time_seconds - 0)
            prediction_stop_watch_canvas.itemconfig(prediction_progress_arc, extent=-prediction_angle)
            prediction_time_value.config(text = self.time_left_format(self.random_character_change_time_seconds)) # Zero-padding for single digit seconds


            # Start game
            while not self.game_terminating:
                ret, frame = self.cap.read()

                # Detect gestures from the frame
                self.detect_gesture(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format from BGR to RGB                
                img = Image.fromarray(frame)  # Create an Image object from the frame
                img = img.resize((1000, 700))  # Adjust the size of the image as needed

                # Create a border around the image
                border_size = 1
                border_image = Image.new("RGB", (img.width + 2*border_size, img.height + 2*border_size), self.border_color)
                border_image.paste(img, (border_size, border_size))

                img = ImageTk.PhotoImage(border_image)  # Create an ImageTk object

                self.canvas.create_image(10, 20, anchor=tk.NW, image=img)  # Display the image on the canvas
                self.canvas.image = img  # Save a reference to the image to prevent it from being garbage collected

                # Calculate time left for change character
                prediction_time_left =int(self.random_character_change_time_seconds) - (int(time.time()) - int(prediction_start_time))
                
                
                prediction_angle = 360 - (360 / self.random_character_change_time_seconds) * (self.random_character_change_time_seconds - prediction_time_left)
                prediction_stop_watch_canvas.itemconfig(prediction_progress_arc, extent=-prediction_angle)
                prediction_time_value.config(text = self.time_left_format(prediction_time_left)) # Zero-padding for single digit seconds


                # Check if it's time to change the random character
                if time.time() - prediction_start_time >= self.random_character_change_time_seconds:
                    self.generate_random_character()

                    # GUI update character
                    # character_value.set(self.random_character)
                    character_value.config(text = str(self.random_character))
                    
                    prediction_start_time = time.time()

                if self.random_character == self.current_prediction_character and self.current_prediction_character is not None:
                    # Calculate time taken for prediction
                    prediction_time_seconds = time.time() - prediction_start_time
                    #update score
                    self.total_score += self.calculate_score(prediction_time_seconds)
                    
                    prediction_start_time = time.time()
                    self.current_prediction_character= None

                    # GUI Total Score
                    total_score_value.config(text = str(self.total_score))
                  
                    
                    # Generate a new random character
                    self.generate_random_character()

                    # GUI update character
                    # print("####### random {} #######".format(self.random_character))
                    character_value.config(text = self.random_character)


                elapsed_time = time.time() - game_start_time
                total_time_left = int(self.total_game_time_seconds) - int(elapsed_time)

                # GUI update total time left 
                # Calculate the angle based on the remaining time
                total_angle = 360 - (360 / self.total_game_time_seconds) * (self.total_game_time_seconds - total_time_left)
                total_stop_watch_canvas.itemconfig(total_progress_arc, extent=-total_angle)
                if(total_time_left<10):
                    total_stop_watch_canvas.itemconfig(total_progress_arc, outline="#F3A8B1")
                    total_time_value.configure(fg="#F3A8B1")
                total_time_value.config(text=self.time_left_format(total_time_left))  # Zero-padding for single digit seconds
                

                # Break the loop if the total game time is reached
                if int(elapsed_time) >= int(self.total_game_time_seconds):
                    self.game_terminating=True
                    continue  
            else:
                # total score
                if self.total_score > self.Max_score_level_1 and game_level==1:
                    self.Max_score_level_1 = self.total_score
                    self.csv_manager.update_high_score(self.user_name,[self.user_name,self.Max_score_level_1,self.Max_score_level_2,self.Max_score_level_3])


                if self.total_score > self.Max_score_level_2 and game_level==2:
                    self.Max_score_level_2 = self.total_score
                    self.csv_manager.update_high_score(self.user_name,[self.user_name,self.Max_score_level_1,self.Max_score_level_2,self.Max_score_level_3])

                
                if self.total_score > self.Max_score_level_3 and game_level==3:
                    self.Max_score_level_3 = self.total_score
                    self.csv_manager.update_high_score(self.user_name,[self.user_name,self.Max_score_level_1,self.Max_score_level_2,self.Max_score_level_3])

                self.total_score = 0

                self.cap.release()  # Release the camera
                self.canvas.delete("all")  # Clear the canvas when the camera feed ends
                self.random_level_window.destroy()  # Close the window
                # Enable buttons:
                self.random_1_button.configure(state='normal')
                self.random_2_button.configure(state='normal')
                self.random_3_button.configure(state='normal')
                self.back_button.configure(state='normal')
            # ---------------------------------

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        frame_width = 250
        frame_height = 140
        padx=5
        pady=5
        lable_size = 10
        component_size = 30
        timer_size = 20
        # component_color = "#FFFFFF"
        component_color ="#C7F2FA"
        # component_color = self.text_color

        # Create the main window
        self.random_level_window = tk.Toplevel()
        # self.random_level_window.geometry("1300x800") 
        width = self.random_level_window.winfo_screenwidth()
        height= self.random_level_window.winfo_screenheight() 
        self.random_level_window.geometry("%dx%d" % (width, height)) 

        


        self.random_level_window.title("Level 1")
        self.random_level_window.configure(bg=self.background_color)

        # Add logo
        self.random_level_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

        # Create a frame to hold the left half content
        left_frame = tk.Frame(self.random_level_window)
        left_frame.configure(bg=self.background_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add logo
        logo = self.load_and_resize_image(self.logo_path, 100)
        logo_label = tk.Label(left_frame, image=logo,bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color)
        logo_label.grid(row=0, column=0, padx=15, pady=15)
        

        total_time_label = tk.Label(left_frame, text="Total Time", font=("Verdana", lable_size, "bold"), bg=self.background_color, fg=self.text_color)
        total_time_label.grid(row=1, column=0, padx=padx, pady=pady)
        total_time_frame = tk.Frame(left_frame, bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color, width=frame_width, height=frame_height)
        total_time_frame.grid(row=2, column=0,padx=padx, pady=pady)
        total_time_frame.grid_propagate(False)  # Prevents the frame from adjusting its size based on content
        total_stop_watch_canvas = tk.Canvas(total_time_frame, width=frame_width-120, height=frame_height-10, bg=self.background_color, highlightthickness=0)
        total_stop_watch_canvas.place(relx=0.5, rely=0.5, anchor="center")
        total_start_angle = 90  # 90 degrees offset to start from the top
        total_end_angle = total_start_angle - 360  # 360 degrees for a full circle
        total_progress_arc = total_stop_watch_canvas.create_arc(10, 10, 120, 120, start=total_start_angle, extent=total_end_angle, outline=component_color, width=5, style="arc")
        total_time_value = tk.Label(total_time_frame, text="00:00", font=("Verdana", timer_size, "bold"), bg=self.background_color, fg=component_color, wraplength=120)
        total_time_value.place(relx=0.5, rely=0.5, anchor="center")

        total_score_label = tk.Label(left_frame, text="Total Score", font=("Verdana", lable_size, "bold"), bg=self.background_color, fg=self.text_color)
        total_score_label.grid(row=1, column=1, padx=padx, pady=pady)
        total_score_frame = tk.Frame(left_frame, bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color, width=frame_width, height=frame_height)
        total_score_frame.grid(row=2, column=1,padx=padx, pady=pady)
        total_score_frame.grid_propagate(False)  # Prevents the frame from adjusting its size based on content
        total_score_value = tk.Label(total_score_frame, text="00", font=("Verdana", component_size, "bold"), bg=self.background_color, fg=component_color, wraplength=120)
        total_score_value.place(relx=0.5, rely=0.5, anchor="center")


        prediction_time_label = tk.Label(left_frame, text="Prediction Time", font=("Verdana", lable_size, "bold"), bg=self.background_color, fg=self.text_color)
        prediction_time_label.grid(row=3, column=0, padx=padx, pady=pady)
        prediction_time_frame = tk.Frame(left_frame, bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color, width=frame_width, height=frame_height)
        prediction_time_frame.grid(row=4, column=0,padx=padx, pady=pady)
        prediction_time_frame.grid_propagate(False)  # Prevents the frame from adjusting its size based on content
        prediction_stop_watch_canvas = tk.Canvas(prediction_time_frame, width=frame_width-120, height=frame_height-10, bg=self.background_color, highlightthickness=0)
        prediction_stop_watch_canvas.place(relx=0.5, rely=0.5, anchor="center")
        prediction_start_angle = 90  # 90 degrees offset to start from the top
        prediction_end_angle = prediction_start_angle - 360  # 360 degrees for a full circle
        prediction_progress_arc = prediction_stop_watch_canvas.create_arc(10, 10, 120, 120, start=prediction_start_angle, extent=prediction_end_angle, outline=component_color, width=5, style="arc")
        prediction_time_value = tk.Label(prediction_time_frame, text="00", font=("Verdana", timer_size, "bold"), bg=self.background_color, fg=component_color, wraplength=120)
        prediction_time_value.place(relx=0.5, rely=0.5, anchor="center")

        character_label = tk.Label(left_frame, text="Character", font=("Verdana", lable_size, "bold"), bg=self.background_color, fg=self.text_color)
        character_label.grid(row=3, column=1, padx=padx, pady=pady)
        character_frame = tk.Frame(left_frame, bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color, width=frame_width, height=frame_height*2.3)
        character_frame.grid(row=4, column=1, rowspan=4, padx=padx, pady=pady)
        character_frame.grid_propagate(False)
        character_value = tk.Label(character_frame, font=("radioland", 150, "bold"), bg=self.background_color, fg=component_color, wraplength=120)
        character_value.place(relx=0.5, rely=0.5, anchor="center")

        max_score_label = tk.Label(left_frame, text="Max Score", font=("Verdana", lable_size, "bold"), bg=self.background_color, fg=self.text_color)
        max_score_label.grid(row=5, column=0, padx=padx, pady=pady)
        max_score_frame = tk.Frame(left_frame, bg=self.background_color, highlightthickness=1, highlightbackground=self.border_color, width=frame_width, height=frame_height)
        max_score_frame.grid(row=6, column=0, padx=padx, pady=pady)
        max_score_frame.grid_propagate(False)
        max_score_value = tk.Label(max_score_frame, text="0", font=("Verdana", component_size, "bold"), bg=self.background_color, fg=component_color, wraplength=120)
        max_score_value.place(relx=0.5, rely=0.5, anchor="center")

         # Add text at the bottom of the right_frame
        # text_label = tk.Label(left_frame, text="Try to predict the character", fg=self.text_color, bg=self.background_color, font=("Verdana", 20, "bold"))
        # # text_label.pack(ipadx=10, ipady=20, anchor=tk.NW, expand=True)
        # text_label.place(relx=0.3, rely=0.9, anchor="center")

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Create a frame to hold the right half content
        right_frame = tk.Frame(self.random_level_window)
        right_frame.configure(bg=self.background_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # # Create a canvas on the right side
        self.canvas = tk.Canvas(right_frame, bg=self.background_color, highlightthickness=5, highlightbackground=self.background_color)
        # self.canvas.place(relx=0.2, rely=0.2,)
        self.canvas.pack(ipadx=10, ipady=80, fill=tk.BOTH, expand=True) 

        # Add text at the bottom of the right_frame
        text_label = tk.Label(right_frame, text='"Unlock your potential and let \nthe game ignite your passion."', fg=self.text_color, bg=self.background_color, font=("Verdana", 20, "bold"))
        # text_label.pack(ipadx=10, ipady=20, anchor=tk.NW, expand=True)
        text_label.place(relx=0.3, rely=0.8, anchor="center")


        #add volume
        # Create a frame to hold the widgets
        volume_frame = tk.Frame(right_frame, bg=self.background_color)
        volume_frame.pack(anchor=tk.SE)
        
        music_level_scale = ctk.CTkSlider(volume_frame, from_=0, to=100, command=self.update_volume)
        music_level_scale.set(self.default_sound_value)
        music_level_scale.grid(row=1, column=1)

        icon_image1 = Image.open("./Game/game_data/Audio.png")
        icon_image1 = icon_image1.resize((15, 15))  # Resize the image if needed
        icon1 = ImageTk.PhotoImage(icon_image1)
        icon_image2 = Image.open("./Game/game_data/NoAudio.png")

        icon_image2 = icon_image2.resize((15, 15))  # Resize the image if needed
        icon2 = ImageTk.PhotoImage(icon_image2)

        icon_label1 = tk.Label(volume_frame, image=icon1, bg=self.background_color)
        icon_label1.grid(row=1, column=2, sticky="s")
        icon_label2 = tk.Label(volume_frame, image=icon2, bg=self.background_color)
        icon_label2.grid(row=1, column=0, sticky="s")


        # Start a thread to update the camera feed
        self.camera_thread = threading.Thread(target=update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # Bind the window closing event to the on_closing function
        self.random_level_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Run the main loop
        self.random_level_window.mainloop()


    def load_and_resize_image(self, path, size):
        image = Image.open(path)
        image = image.resize((size, size), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def learn_level(self):

        self.game_terminating = False

        def on_closing():
            self.game_terminating = True
            
        def update_camera():
            # disabled buttons:
            self.learn_button.configure(state='disabled')
            self.Start_Play_button.configure(state='disabled')
            self.open_score_button.configure(state='disabled')

            # fix camera
            ret, frame = self.cap.read()
            if not ret:
                self.cap = cv2.VideoCapture(0)

            # game logic
            # --------------------------------
            while not self.game_terminating:
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Detect gestures from the frame
                self.detect_gesture(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format from BGR to RGB
                img = Image.fromarray(frame)  # Create an Image object from the frame
                img = img.resize((1000, 700))  # Adjust the size of the image as needed

                # Create a border around the image
                border_size = 1
                border_image = Image.new("RGB", (img.width + 2*border_size, img.height + 2*border_size), self.border_color)
                border_image.paste(img, (border_size, border_size))

                img = ImageTk.PhotoImage(border_image)  # Create an ImageTk object

                self.canvas.create_image(10, 20, anchor=tk.NW, image=img)  # Display the image on the canvas
                self.canvas.image = img  # Save a reference to the image to prevent it from being garbage collected
            else:
                self.canvas.delete("all")  # Clear the canvas when the camera feed ends
                self.cap.release()  # Release the camera
                self.learn_level_window.destroy()  # Close the window
                # Enable buttons:
                self.learn_button.configure(state='normal')
                self.Start_Play_button.configure(state='normal')
                self.open_score_button.configure(state='normal')

             # --------------------------------
     
        # Create the main window
        self.learn_level_window = tk.Toplevel()

        # self.learn_level_window.geometry("1300x800")
        width = self.learn_level_window.winfo_screenwidth()
        height= self.learn_level_window.winfo_screenheight() 

        self.learn_level_window.geometry("%dx%d" % (width, height)) 

        self.learn_level_window.title("How To Play")
        self.learn_level_window.configure(bg=self.background_color)

        # Create a frame to hold the left half content
        left_frame = tk.Frame(self.learn_level_window)
        left_frame.configure(bg=self.background_color)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add logo
        image_path = self.logo_path    
        my_image = ctk.CTkImage(dark_image=Image.open(image_path),size=(250, 250))
        image_label = ctk.CTkLabel(left_frame, image=my_image, text="") 
        image_label.place(relx=0.2, rely=0.05)

        # Add logo
        self.learn_level_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

        # Add char_signs image
        char_signs = "./Game/game_data/char_temp.png"    
        my_image = ctk.CTkImage(dark_image=Image.open(char_signs),size=(480, 635))
        char_signs_title_label = ctk.CTkLabel(left_frame, image=my_image, text="") 
        char_signs_title_label.place(relx=0.0, rely=0.3)

        # Create a frame to hold the right half content
        right_frame = tk.Frame(self.learn_level_window)
        right_frame.configure(bg=self.background_color)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # # Create a canvas on the right side
        self.canvas = tk.Canvas(right_frame, bg=self.background_color, highlightthickness=5, highlightbackground=self.background_color)
        self.canvas.pack(ipadx=10, ipady=80, fill=tk.BOTH, expand=True) 

        # Add text at the bottom of the right_frame
        text_label = ctk.CTkLabel(right_frame,text='"The difference between try and triumph is\n just a little umph!"', font=('Comic Sans MS',30,"bold",))
        text_label.place(relx=0.4, rely=0.85,anchor=ctk.CENTER)

         #add volume
        # Create a frame to hold the widgets
        volume_frame = tk.Frame(right_frame, bg=self.background_color)
        volume_frame.pack(anchor=tk.SE)
        
        music_level_scale = ctk.CTkSlider(volume_frame, from_=0, to=100, command=self.update_volume)
        music_level_scale.set(self.default_sound_value)
        music_level_scale.grid(row=1, column=1)
        icon_image1 = Image.open("./Game/game_data/Audio.png")
        icon_image1 = icon_image1.resize((15, 15))  # Resize the image if needed
        icon1 = ImageTk.PhotoImage(icon_image1)
        icon_image2 = Image.open("./Game/game_data/NoAudio.png")
        icon_image2 = icon_image2.resize((15, 15))  # Resize the image if needed
        icon2 = ImageTk.PhotoImage(icon_image2)
        icon_label1 = tk.Label(volume_frame, image=icon1, bg=self.background_color)
        icon_label1.grid(row=1, column=2, sticky="s")
        icon_label2 = tk.Label(volume_frame, image=icon2, bg=self.background_color)
        icon_label2.grid(row=1, column=0, sticky="s")

        # Start a thread to update the camera feed
        self.camera_thread = threading.Thread(target=update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # Bind the window closing event to the on_closing function
        self.learn_level_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Run the main loop
        self.learn_level_window.mainloop()


    def open_game_menu(self):
        # Clear the main menu window
        if  self.Main_window != None:
            self.Main_window.destroy()
            

        user_scores = self.csv_manager.get_user_scores(self.user_name)
        self.Max_score_level_1 = user_scores["Max_score_level_1"]
        self.Max_score_level_2 = user_scores["Max_score_level_2"]
        self.Max_score_level_3 = user_scores["Max_score_level_3"]

        # Create the play game menu window
        self.game_menu_window = tk.Tk()
        self.game_menu_window.title("Play Game")
        self.game_menu_window.title("Sign-Saga")
        # self.game_menu_window.geometry("1000x1000") 
        self.game_menu_window.configure(bg=self.background_color)

        width = self.game_menu_window.winfo_screenwidth()
        height= self.game_menu_window.winfo_screenheight() 
        self.game_menu_window.geometry("%dx%d" % (width, height))

        image_path = "BackgroundImage.png" 
        my_image = ctk.CTkImage(dark_image=Image.open(image_path),size=(width,height))
        image_label = ctk.CTkLabel(self.game_menu_window, image=my_image, text="") 
        image_label.pack()        

        # Add logo icon
        self.game_menu_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

        def level_1():
            self.random_level(total_time=300, character_change_time=20,game_level=1)

        def level_2():
            self.random_level(total_time=300, character_change_time=10,game_level=2)

        def level_3():
            self.random_level(total_time=300, character_change_time=5,game_level=3)

      
        # Add a button to start the game
        self.random_1_button = ctk.CTkButton(self.game_menu_window, text="⭐ Easy ⭐",height=50, width=350,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=level_1)
        self.random_1_button.place(relx=0.5, rely=0.6, anchor=ctk . CENTER)

        # Add a button to start the game
        self.random_2_button = ctk.CTkButton(self.game_menu_window, text="⭐⭐ Medium ⭐⭐",height=50, width=350,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=level_2)
        self.random_2_button.place(relx=0.5, rely=0.7, anchor=ctk . CENTER)

        # Add a button to start the game
        self.random_3_button = ctk.CTkButton(self.game_menu_window, text="⭐⭐⭐ Brutal ⭐⭐⭐",height=50, width=350,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=level_3)
        self.random_3_button.place(relx=0.5, rely=0.8, anchor=ctk . CENTER)

        # Add a "Back" button to return to the main menu
        self.back_button = ctk.CTkButton(self.game_menu_window, text="⬅️Back",height=50, width=60,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25, "bold"),command=self.open_main_menu)
        self.back_button.place(relx=0.05, rely=0.05, anchor=ctk . CENTER)

         #add volume
        # Create a frame to hold the widgets
        volume_frame = tk.Frame(self.game_menu_window, bg='black',)
        volume_frame.place(relx=0.9, rely=.95, anchor=ctk.CENTER)

        music_level_scale = ctk.CTkSlider(volume_frame, from_=0, to=100, command=self.update_volume)
        music_level_scale.set(self.default_sound_value)
        music_level_scale.grid(row=1, column=1)
        icon_image1 = Image.open("./Game/game_data/Audio.png")
        icon_image1 = icon_image1.resize((15, 15))  # Resize the image if needed
        icon1 = ImageTk.PhotoImage(icon_image1)
        icon_image2 = Image.open("./Game/game_data/NoAudio.png")
        icon_image2 = icon_image2.resize((15, 15))  # Resize the image if needed
        icon2 = ImageTk.PhotoImage(icon_image2)
        icon_label1 = tk.Label(volume_frame, image=icon1, bg=self.background_color)
        icon_label1.grid(row=1, column=2, sticky="s")
        icon_label2 = tk.Label(volume_frame, image=icon2, bg=self.background_color)
        icon_label2.grid(row=1, column=0, sticky="s")

        # Run the play game menu loop
        self.game_menu_window.mainloop()


    def open_score_window(self):
        # Clear the main menu window
        if  self.Main_window != None:
            self.Main_window.destroy()
            self.Main_window = None

        def sort_score(tab):
            scores_obj = self.csv_manager.get_top_users()
            # tab = event.widget.tab('current')['text']

            if tab== 'level 1 scores':
                    level_data=scores_obj['Max_score_level_1']
                    clean_table(tab)
                    n=0
                    # sorted(level_data, key=lambda x:x[1]) ////////////////////////////////////////////////////********
                    for data in level_data :
                        if n %2==0:
                            table.insert(parent='', index=0 , values=data, tags=('evenrow',))
                        else :
                            table.insert(parent='', index=0 , values=data , tags=('oddrow',))
                        n+=1
            elif tab== 'level 2 scores' :
                level_data=scores_obj['Max_score_level_2']
                clean_table(tab)
                n=0
                for data in level_data :
                    if n %2==0:
                        table1.insert(parent='', index=0 , values=data,tags=('evenrow',))
                    else:
                        table1.insert(parent='', index=0 , values=data , tags=('oddrow',))
                    n+=1

            elif tab== 'level 3 scores' :
                level_data=scores_obj['Max_score_level_3']
                clean_table(tab)
                n=0
                for data in level_data :
                    if n %2==0:
                        table2.insert(parent='', index=0 , values=data,tags=('evenrow',))
                    else:
                        table2.insert(parent='', index=0 , values=data,tags=('oddrow',))
                    n+=1

        def clean_table(tab) :
                if tab== 'level 1 scores':
                    for item in table.get_children():
                        table.delete(item)
                    
                elif tab== 'level 2 scores' :
                    for item in table1.get_children():
                        table1.delete(item)
                
                elif tab== 'level 3 scores':
                    for item in table2.get_children():
                        table2.delete(item)

        # Create the score window
        self.score_window = tk.Tk()
        self.score_window.title("Score Window")
        width = self.score_window.winfo_screenwidth()
        height= self.score_window.winfo_screenheight() 
        self.score_window.geometry("%dx%d" % (width, height))
        # self.score_window.geometry("1000x1000") 
        self.score_window.configure(bg=self.background_color)

        
        self.canvas = tk.Canvas(self.score_window, bg=self.background_color, highlightbackground=self.background_color) #**
        self.canvas.place(relx=0.15, rely=0.03,)
        # self.canvas.pack(ipadx=100, ipady=505, fill=tk.BOTH, expand=True) #**

        # image_path = "293347565.jpg"
        # my_image = ctk.CTkImage(dark_image=Image.open(image_path),size=(width,height))
        # image_label = ctk.CTkLabel(self.score_window, image=my_image, text="") 
        # image_label.pack()

        # Add logo icon
        self.score_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

        noteBook=ctk.CTkTabview(self.canvas, command=lambda :sort_score(noteBook.get()),
                                corner_radius=20,
                                border_width=5,
                                border_color="white")    # ttk.Notebook(canvas_widget)
        
        noteBook.pack()

        # Add some style
        style = ttk.Style()
        #Pick a theme
        style.theme_use("default")
        # Configure our treeview colors
        
        style.configure("Treeview",
                        background="#2a2d2e", 
                        foreground="white",
                        rowheight=40,
                        # corner_radius = 50,
                        fieldbackground=self.background_color,
                        bordercolor=self.background_color,
                        borderwidth=0,
                        # image = "293347565.jpg",
                        font=("Comic Sans MS",15,"bold"))
        
        style.map('Treeview', background=[('selected', '#28393a')])

        style.configure("Treeview.Heading",
                        background=self.background_color,
                        foreground=self.background_color,
                        relief="flat")
        style.map("Treeview.Heading",
                background=[('active', '#3484F0')],
                )
        
        # --------------------------------------------------------------------
        tab1=noteBook.add('level 1 scores')       #ttk.Frame(noteBook ,style='Custom.TFrame')
        noteBook.set('level 1 scores')
        
        table=ttk.Treeview(tab1, column=('name','score'))

        table.column('#0', width=0 , anchor=tk.CENTER)
        table.column('name' , anchor=tk.W , width=600,)
        table.column('score' , anchor=tk.W , width=600)

        # table.heading('name', text='player name',anchor=tk.W)
        # table.heading('score', text='level 1 scores',anchor=tk.W)
        table.pack(fill='both' , expand=True , anchor=tk.W)
        sort_score('level 1 scores')

        # ----------------------------------------
        tab2=noteBook.add('level 2 scores')             #ttk.Frame(noteBook)

        table1=ttk.Treeview(tab2 ,column=('name', 'score'))

        table1.column('#0', width=0 , anchor=tk.W)
        table1.column('name' , anchor=tk.W , width=600)
        table1.column('score' , anchor=tk.W , width=600)

        # table1.heading('name', text='player name')
        # table1.heading('score', text='level 2 scores')
        table1.pack(fill='both' , expand=True , anchor=tk.W)
        # --------------------------------------
        tab3=noteBook.add('level 3 scores',)                         #ttk.Frame(noteBook)

        table2=ttk.Treeview(tab3 , column=('name', 'score'))

        table2.column('#0', width=0 , anchor=tk.W)
        table2.column('name' , anchor=tk.W , width=600)
        table2.column('score' , anchor=tk.W , width=600)

        # table2.heading('name', text='player name')
        # table2.heading('score', text='level 3 scores')
        table2.pack(fill='both' , expand=True , anchor=tk.W)
     
        table.tag_configure('oddrow', background="#343638")
        table.tag_configure('evenrow', background="#146C94")

        table1.tag_configure('oddrow', background="#343638")
        table1.tag_configure('evenrow', background="#146C94")

        table2.tag_configure('oddrow', background="#343638")
        table2.tag_configure('evenrow', background="#146C94")

        # Add logo icon
        # self.score_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

        self.back_button_2 = ctk.CTkButton(self.score_window, text="⬅️Back",height=50, width=60,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25, "bold"),command=self.open_main_menu)
        self.back_button_2.place(relx=0.05, rely=0.05, anchor=ctk . CENTER)

         #add volume
        # Create a frame to hold the widgets
        volume_frame = tk.Frame(self.score_window, bg='#28393a',)
        volume_frame.place(relx=0.9, rely=.95, anchor=ctk.CENTER)
        # volume_frame.pack(anchor=tk.SE)
        
        music_level_scale = ctk.CTkSlider(volume_frame, from_=0, to=100, command=self.update_volume)
        music_level_scale.set(self.default_sound_value)
        music_level_scale.grid(row=1, column=1)
        icon_image1 = Image.open("./Game/game_data/Audio.png")
        icon_image1 = icon_image1.resize((15, 15))  # Resize the image if needed
        icon1 = ImageTk.PhotoImage(icon_image1)
        icon_image2 = Image.open("./Game/game_data/NoAudio.png")
        icon_image2 = icon_image2.resize((15, 15))  # Resize the image if needed
        icon2 = ImageTk.PhotoImage(icon_image2)
        icon_label1 = tk.Label(volume_frame, image=icon1, bg=self.background_color)
        icon_label1.grid(row=1, column=2, sticky="s")
        icon_label2 = tk.Label(volume_frame, image=icon2, bg=self.background_color)
        icon_label2.grid(row=1, column=0, sticky="s")

        # Run the play game menu loop
        self.score_window.mainloop()
    #---------------------------------------    

    def open_main_menu(self):
        # Clear the game menu window
        if self.game_menu_window != None:
            self.game_menu_window.destroy()
            self.game_menu_window = None

        if self.score_window != None:
            self.score_window.destroy()
            self.score_window = None

        def update_selected_user(choice):
            self.user_name = str(choice)
            nameEntry.configure(placeholder_text=self.user_name)
            selected_user_var.set(self.user_name)
            

        def populate_user_menu():
            users = self.csv_manager.get_users()
            users.pop(0)
            self.user_dropdown.configure(values=users)
            nameEntry.configure(placeholder_text=self.user_name)
        
      
        # Function to open the game menu
        def open_game_validation():
            user_name = nameEntry.get()
            if user_name:
                if user_name in self.csv_manager.get_users():
                    self.user_name = user_name
                    selected_user_var.set(self.user_name)
                    if defult_name == user_name:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game without user")
                    else:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game menu with {user_name} user ")
                    if result == 'yes':
                        self.open_game_menu()
        
                else:
                    self.csv_manager.add_user(user_name)
                    selected_user_var.set(user_name)  # Set the newly added name as the selected user
                    self.user_name = user_name
                    populate_user_menu()
                    if defult_name == user_name:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game without user")
                    else:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game menu with {user_name} user ")
                    if result == 'yes':
                        self.open_game_menu()
                      
            else:
                selected_user = selected_user_var.get()
                if selected_user:
                    self.user_name = selected_user
                    nameEntry.configure(placeholder_text=selected_user)
                    if defult_name == selected_user:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game without user")
                    else:
                        result = tk.messagebox.askquestion("Game Menu", f"are you sure you want to open the game menu with {selected_user} user ")
                    if result == 'yes':
                        self.open_game_menu()
                else:
                    tk.messagebox.showwarning("Error", "Please select a user.")
                    
        # Create a UserCSVManager instance
        self.csv_manager = UserCSVManager(self.user_data_path)

        # Create the main window
        self.Main_window = tk.Tk()
        self.Main_window.title("Sign-Saga")
        # self.Main_window.geometry("1000x1000") 
        self.Main_window.configure(bg=self.background_color)

        width = self.Main_window.winfo_screenwidth()
        height= self.Main_window.winfo_screenheight() 
        self.Main_window.geometry("%dx%d" % (width, height)) 

        image_path = "BackgroundImage.png"
        my_image = ctk.CTkImage(dark_image=Image.open(image_path),size=(width,height))
        image_label = ctk.CTkLabel(self.Main_window, image=my_image, text="") 
        image_label.pack()

        # Add logo icon
        self.Main_window.iconphoto(False, tk.PhotoImage(file = self.logo_path))

    #    Create a dropdown menu to display the users
       # Create a dropdown menu to display the users
        selected_user_var = ctk.StringVar(self.Main_window)
        users = self.csv_manager.get_users()
        if self.user_name == None:
            self.user_name = users.pop(0)
            defult_name = self.user_name
        else:
            defult_name = users.pop(0)

        selected_user_var.set(self.user_name)
        self.user_dropdown = ctk.CTkOptionMenu(self.Main_window, variable=selected_user_var, values=users,height=40,width=200,font=("Verdana", 20,"bold"),fg_color="#146C94",command=update_selected_user)
        self.user_dropdown.place(relx=0.44, rely=0.6, anchor=ctk . CENTER)

        nameEntry = ctk.CTkEntry(self.Main_window,placeholder_text=self.user_name,height=40, width=200,corner_radius=10,font=("Verdana", 20,"bold")) 
        nameEntry.place(relx=0.56, rely=0.6, anchor=ctk . CENTER)
    
        # Add a button to open play_game_menu
        self.Start_Play_button = ctk.CTkButton(self.Main_window, text="⚔️ Start Play ⚔️",height=50, width=400,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=open_game_validation)
        self.Start_Play_button.place(relx=0.5, rely=0.7, anchor=ctk . CENTER)

        # Add a button to open How to Play
        self.learn_button = ctk.CTkButton(self.Main_window, text="⚓ How to Play",height=50, width=400,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=self.learn_level)
        self.learn_button.place(relx=0.5, rely=0.8, anchor=ctk . CENTER)

        # Add a button to open top score
        self.open_score_button = ctk.CTkButton(self.Main_window, text="👑 Top scores",height=50, width=400,fg_color="#146C94",hover_color="#28393a",font=("Verdana", 25,"bold"),command=self.open_score_window)
        self.open_score_button.place(relx=0.5, rely=0.9, anchor=ctk . CENTER)

        #add volume
        # Create a frame to hold the widgets
        volume_frame = tk.Frame(self.Main_window, bg='black',)
        volume_frame.place(relx=0.9, rely=.95, anchor=ctk.CENTER)
        # volume_frame.pack(anchor=tk.SE)
        
        music_level_scale = ctk.CTkSlider(volume_frame, from_=0, to=100, command=self.update_volume)
        music_level_scale.set(self.default_sound_value)
        music_level_scale.grid(row=1, column=1)
        icon_image1 = Image.open("./Game/game_data/Audio.png")
        icon_image1 = icon_image1.resize((15, 15))  # Resize the image if needed
        icon1 = ImageTk.PhotoImage(icon_image1)
        icon_image2 = Image.open("./Game/game_data/NoAudio.png")
        icon_image2 = icon_image2.resize((15, 15))  # Resize the image if needed
        icon2 = ImageTk.PhotoImage(icon_image2)
        icon_label1 = tk.Label(volume_frame, image=icon1, bg=self.background_color)
        icon_label1.grid(row=1, column=2, sticky="s")
        icon_label2 = tk.Label(volume_frame, image=icon2, bg=self.background_color)
        icon_label2.grid(row=1, column=0, sticky="s")

        # Run the main loop
        self.Main_window.mainloop()

    def start_game(self):
        # # Load and play the audio track
        pygame.mixer.music.load(self.sound_path)
        pygame.mixer.music.play(-1)  # Set -1 to play the track in a loop indefinitely
        self.update_volume(self.default_sound_value)
        self.open_main_menu()
