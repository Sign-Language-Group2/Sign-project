import pickle
import random
import cv2
import mediapipe as mp
import numpy as np
import time

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

    def destroy_game_windows(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def learn_level(self):
        #gui start game


        #start game
        while True:
            ret, frame = self.cap.read()
            # Detect gestures from the frame
            self.detect_gesture(frame)
            cv2.imshow('frame', frame)

            # Break the loop if the user presses q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #if gui x brak

        #gui print total score



        time.sleep(5)
        self.destroy_game_windows()

    def calculate_score(self,prediction_time_seconds):
        if prediction_time_seconds < 2:
            return 3
        else:
            return 1

    def random_level(self, total_time=20, character_change_time=5):

        # Set total game time and character change time
        self.total_game_time_seconds = total_time
        self.random_character_change_time_seconds = character_change_time

        # Start the game timer
        game_start_time = time.time()
        prediction_start_time = time.time()
        prediction_time_seconds = 0

        # GUI start game


        # Generate initial random character
        self.generate_random_character()

        # GUI initial character and Total score
        print("####### random {} #######".format(self.random_character))
        print(self.total_score)

        
        # Start game
        while True:
            ret, frame = self.cap.read()

            # Detect gestures from the frame
            self.detect_gesture(frame)
            cv2.imshow('frame', frame)

            # Calculate time left for change character
            prediction_time_left =int(self.random_character_change_time_seconds) - (int(time.time()) - int(prediction_start_time))
            
            # GUI update time left for change character
            #print(prediction_time_left)



            # Check if it's time to change the random character
            if time.time() - prediction_start_time >= self.random_character_change_time_seconds:
                self.generate_random_character()

                # GUI update character
                print("####### random {} #######".format(self.random_character))

                prediction_start_time = time.time()

            if self.random_character == self.current_prediction_character and self.current_prediction_character is not None:
                # Calculate time taken for prediction
                prediction_time_seconds = time.time() - prediction_start_time
                #update score
                self.total_score += self.calculate_score(prediction_time_seconds)
                
                prediction_start_time = time.time()
                self.current_prediction_character= None

                # GUI Total Score
                print(self.total_score)
                
                # Generate a new random character
                self.generate_random_character()

                # GUI update character
                print("####### random {} #######".format(self.random_character))



                


            # Break the loop if the user presses q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Break the loop if the total game time is reached
            elapsed_time = time.time() - game_start_time
            if int(elapsed_time) >= int(self.total_game_time_seconds):
                break

            total_time_left = int(self.total_game_time_seconds) - int(elapsed_time)
            #print(total_time_left)
            # GUI update total time left 



            # if GUI x break

        # GUI print total score

        print("Total Score:", self.total_score)

        # time.sleep(5)
        self.destroy_game_windows()

    def start_game(self):

        self.random_level()

        




