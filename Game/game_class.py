import sys
sys.path.insert(1, r'c:\users\myips\desktop\signproject\sign-project\.venv\lib\site-packages')

import pickle
import random
import cv2
import mediapipe as mp
import numpy as np
import time

import customtkinter as ctk
import tkinter as tk
import cv2
from PIL import ImageTk, Image
import threading

from Game.rounded_button_class import RoundedButton




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


        self.game_terminating = False

        self.image_tk = None 
        self.opening_image_tk= None

        self.game_menu_window = None
        self.Main_window = None
        self.learn_level_window = None
        self.random_level_window = None

        self.canvas= None
        self.camera_thread= None

        self.learn_button = None
        self.Start_Play_button = None

        self.random_1_button = None
        self.back_button = None


        self.widgets = {
                        "logo": [],
                        "logo_main": [],    
                        "play_button": [],
                        "how_to_play_button": [],
                        "char_signs": [],
                        "return_home_button": [],
                        }

    def generate_random_character(self):
        # Randomly choose a character from the labels_dict values
        self.random_character = random.choice(list(self.labels_dict.values()))

    def read_frame(self):
        # Read a frame from the video capture object
        ret, frame = self.cap.read()
        return frame

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



    def random_level(self, total_time=20, character_change_time=5,game_level=1):

        self.game_terminating = False

        def on_closing():
            self.game_terminating = True
            

        def update_camera():
            # disabled buttons:
            self.random_1_button.config(state='disabled')
            self.back_button.config(state='disabled')

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
                max_score_value.set(self.Max_score_level_1)
            character_value.set(self.random_character)
            total_score_value.set(str(self.total_score))
            total_time_value.set(str(self.total_game_time_seconds))
            prediction_time_value.set(str(self.random_character_change_time_seconds))

            # Start game
            while not self.game_terminating:
                ret, frame = self.cap.read()

                # Detect gestures from the frame
                self.detect_gesture(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format from BGR to RGB
                img = Image.fromarray(frame)  # Create an Image object from the frame
                img = img.resize((400, 400))  # Adjust the size of the image as needed
                img = ImageTk.PhotoImage(img)  # Create an ImageTk object

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)  # Display the image on the canvas
                self.canvas.image = img  # Save a reference to the image to prevent it from being garbage collected
                
                # Calculate time left for change character
                prediction_time_left =int(self.random_character_change_time_seconds) - (int(time.time()) - int(prediction_start_time))
                
                # GUI update time left for change character
                #print(prediction_time_left)
                prediction_time_value.set(str(prediction_time_left))

                # Check if it's time to change the random character
                if time.time() - prediction_start_time >= self.random_character_change_time_seconds:
                    self.generate_random_character()

                    # GUI update character
                    #print("####### random {} #######".format(self.random_character))
                    character_value.set(self.random_character)
                    
                    prediction_start_time = time.time()

                if self.random_character == self.current_prediction_character and self.current_prediction_character is not None:
                    # Calculate time taken for prediction
                    prediction_time_seconds = time.time() - prediction_start_time
                    #update score
                    self.total_score += self.calculate_score(prediction_time_seconds)
                    
                    prediction_start_time = time.time()
                    self.current_prediction_character= None

                    # GUI Total Score
                    #print(self.total_score)
                    total_score_value.set(str(self.total_score))
                  
                    
                    # Generate a new random character
                    self.generate_random_character()

                    # GUI update character
                    # print("####### random {} #######".format(self.random_character))
                    character_value.set(self.random_character)


                elapsed_time = time.time() - game_start_time
                total_time_left = int(self.total_game_time_seconds) - int(elapsed_time)
                #print(total_time_left)
                # GUI update total time left 
                total_time_value.set(str(total_time_left))

                # Break the loop if the total game time is reached
                if int(elapsed_time) >= int(self.total_game_time_seconds):
                    self.game_terminating=True
                    continue  
            else:
                # reset total score
                if self.total_score > self.Max_score_level_1:
                    self.Max_score_level_1 = self.total_score
                self.total_score = 0
                # 
                self.cap.release()  # Release the camera
                self.canvas.delete("all")  # Clear the canvas when the camera feed ends
                self.random_level_window.destroy()  # Close the window
                # Enable buttons:
                self.random_1_button.config(state='normal')
                self.back_button.config(state='normal')
            # ---------------------------------

            

        # Create the main window
        self.random_level_window = tk.Toplevel()
        self.random_level_window.geometry("800x600")  # Set the window size

        # Set the title and logo
        self.random_level_window.title("My Window")

        # Create a frame to hold the left half content
        left_frame = tk.Frame(self.random_level_window)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create StringVar variables for the component values
        max_score_value = tk.StringVar()
        character_value = tk.StringVar()
        total_score_value = tk.StringVar()
        total_time_value = tk.StringVar()
        prediction_time_value = tk.StringVar()

        
        # Add label for character
        max_score_label = tk.Label(left_frame, text="Max score:")
        max_score_label.pack()
        max_score_label = tk.Label(left_frame, textvariable= max_score_value)
        max_score_label.pack()

        # Add label for character
        character_label = tk.Label(left_frame, text="Character:")
        character_label.pack()
        character_display = tk.Label(left_frame, textvariable=character_value)
        character_display.pack()

        # Add label for total score
        total_score_label = tk.Label(left_frame, text="Total Score:")
        total_score_label.pack()
        total_score_display = tk.Label(left_frame, textvariable=total_score_value)
        total_score_display.pack()

        # Add label for total time left
        total_time_label = tk.Label(left_frame, text="Total Time Left:")
        total_time_label.pack()
        total_time_display = tk.Label(left_frame, textvariable=total_time_value)
        total_time_display.pack()

        # Add label for prediction time left
        prediction_time_label = tk.Label(left_frame, text="Prediction Time Left:")
        prediction_time_label.pack()
        prediction_time_display = tk.Label(left_frame, textvariable=prediction_time_value)
        prediction_time_display.pack()

        # Create a frame to hold the right half content
        right_frame = tk.Frame(self.random_level_window)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a canvas on the right side
        self.canvas = tk.Canvas(right_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Start a thread to update the camera feed
        self.camera_thread = threading.Thread(target=update_camera)
        self.camera_thread.daemon = True
        self.camera_thread.start()

        # Bind the window closing event to the on_closing function
        self.random_level_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Run the main loop
        self.random_level_window.mainloop()



    def learn_level(self):

        self.game_terminating = False

        def on_closing():
            self.game_terminating = True
            
        def update_camera():
            # disabled buttons:
            self.learn_button.config(state='disabled')
            self.Start_Play_button.config(state='disabled')

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
                img = img.resize((400, 400))  # Adjust the size of the image as needed
                img = ImageTk.PhotoImage(img)  # Create an ImageTk object

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)  # Display the image on the canvas
                self.canvas.image = img  # Save a reference to the image to prevent it from being garbage collected
            else:
                self.canvas.delete("all")  # Clear the canvas when the camera feed ends
                self.cap.release()  # Release the camera
                self.learn_level_window.destroy()  # Close the window
                # Enable buttons:
                self.learn_button.config(state='normal')
                self.Start_Play_button.config(state='normal')

             # --------------------------------

        

        # Create the main window
        self.learn_level_window = tk.Toplevel()
        self.learn_level_window.geometry("1000x1000")  # Set the window size
        
        # Set the title and logo
        self.learn_level_window.title("How To Play")
        self.learn_level_window.iconbitmap(r'Game\game_data\Red And Yellow Illustration Rock Music (1).ico')

        self.learn_level_window.configure(bg="blue")

        # Create a frame to hold the left half content
        left_frame = tk.Frame(self.learn_level_window)
        left_frame.pack(padx=10,pady=10,side=tk.LEFT, fill=tk.BOTH, expand=True)
        
    
        # Add an logo
        # image = Image.open("./Game/game_data/Red And Yellow Illustration Rock Music (1).png")
        # image = image.resize((250, 250), Image.LANCZOS)
        # logo = ImageTk.PhotoImage(image)
        # logo_label = tk.Label(left_frame, image=logo, highlightthickness=0, borderwidth=0)
        # logo_label.image = logo
        # self.widgets["logo"].append(logo_label)
        # logo_label.pack(pady=(0, 0))  # Centered vertically with 50 pixels padding at the top


        # Add an image on the left side
        char_signs = Image.open("./Game/game_data/chars.png")
        char_signs_title = ImageTk.PhotoImage(char_signs)
        char_signs_title_label = tk.Label(left_frame, image=char_signs_title, highlightthickness=0, borderwidth=0,bg="blue")
        char_signs_title_label.image = char_signs_title
        self.widgets['char_signs'].append(char_signs_title_label)
        char_signs_title_label.place(x=10, y=200 ) 

   

        # Create a frame to hold the right half content
        right_frame = tk.Frame(self.learn_level_window)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a canvas on the right side
        self.canvas = tk.Canvas(right_frame, bg="#161219")
        self.canvas.pack(padx=10,pady=10,fill=tk.BOTH, expand=True)

    
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

        # Create the play game menu window
        self.game_menu_window = tk.Tk()
        self.game_menu_window.geometry("800x600")
        self.game_menu_window.title("Play Game")
        self.game_menu_window.iconbitmap(r'Game\game_data\Red And Yellow Illustration Rock Music (1).ico')

        # Add the game content
        game_label = tk.Label(self.game_menu_window, text="Game Content")
        game_label.pack()

        # Add a button to start the game
        
        self.random_1_button = tk.Button(self.game_menu_window, text="Start Game", command=self.random_level)
        # self.random_1_button = tk.Button(self.game_menu_window, text="Start Game", command=self.random_level,
        #                                  bg="#3366CC", fg="#FFFFFF", bd=0,
        #                                  activebackground="#224499", activeforeground="#FFFFFF",
        #                                  padx=10, pady=5, font=("Arial", 12, "bold"))
        
        self.random_1_button.pack()

        # Add a "Back" button to return to the main menu
        self.back_button = tk.Button(self.game_menu_window, text="Back", command=self.open_main_menu)
        self.back_button.pack()

        # Run the play game menu loop
        self.game_menu_window.mainloop()


    def open_main_menu(self):
        # Clear the game menu window
        if self.game_menu_window != None:
            self.game_menu_window.destroy()


        # Create the main window
        self.Main_window = tk.Tk()
        self.Main_window.geometry("1000x1000")  # Set the window size
        

        # Set the title and logo
        self.Main_window.title("SignSaga")
        self.Main_window.iconbitmap(r'Game\game_data\Red And Yellow Illustration Rock Music (1).ico')
        self.Main_window.configure(bg="#161219")


        # Display logos
        image = Image.open("./Game/game_data/Red And Yellow Illustration Rock Music (1).png")
        image = image.resize((250, 250), Image.LANCZOS)
        logo = ImageTk.PhotoImage(image)
        logo_label = tk.Label(self.Main_window, image=logo, highlightthickness=0, borderwidth=0)
        logo_label.image = logo
        self.widgets["logo"].append(logo_label)
        logo_label.pack(pady=(0, 0))  # Centered vertically with 50 pixels padding at the top



        # # Display main logos
        # image_main = Image.open("./Game/game_data/Screenshot from 2023-05-27 04-22-49.png")
        # logo_main = ImageTk.PhotoImage(image_main)
        # logo_main_label = tk.Label(self.Main_window, image=logo_main, highlightthickness=0, borderwidth=0)
        # logo_main_label.image = logo_main
        # self.widgets["logo_main"].append(logo_main_label)
        # logo_main_label.pack(pady=(50, 0))  # Centered vertically with 50 pixels padding at the top



        # Add a button to open play_game_menu
        self.Start_Play_button = RoundedButton(self.Main_window, text="New Game",command=self.open_game_menu)
        self.widgets["play_button"].append(self.Start_Play_button)
        self.Start_Play_button.pack(pady=(50, 0))  # Centered vertically with 50 pixels padding at the top

        # Add a button to open How to Play
        self.learn_button = RoundedButton(self.Main_window, text="How to Play",command=self.learn_level)
        self.widgets["how_to_play_button"].append(self.learn_button)
        self.learn_button.pack(pady=(50, 0))  # Centered vertically with 


        # Run the main loop
        self.Main_window.mainloop()

    def start_game(self):
        self.open_main_menu()



        




