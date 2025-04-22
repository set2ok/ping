import tensorflow as tf
import numpy as np
import math

from pygame.midi import Input


class Bot():
    def __init__(self, player, oponents, balls,input_shape=(102,), model_path=None):
        # inputs: player:[[x,y], [width, height], [speed], 4*[bound]]:13
        # 4 * ball:[[x,y], [cos, sin], [speed, radius]]: 4 * 6
        # 5* oponent [[x,y], [width, height], [speed], 4*[bound]] 5* 13
        # -where the opondent it controls are att fist postition

        self.player = player
        self.oponents = oponents
        self.balls = balls

        self.how_often = 10
        self.call_count = 0
        self.last_move = None
        self.past_states= []

        self.max_balls = 4
        self.max_oponents = 5

        self.input_shape = input_shape
        self.model = self.build_model() if model_path is None else tf.keras.models.load_model(model_path)
    def build_model(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(16, activation = "relu"),
            tf.keras.layers.BatchNormalization(),  # Normalizes activations
            tf.keras.layers.Dropout(1/16),  # Prevents overfitting
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(3/32),
            tf.keras.layers.Dense(self.max_oponents, activation='tanh')  # Output range (-1 to 1) for movement for every oponent
        ])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def create_input(self):
        input_data = []
        input_data += self.player_input()
        input_data += self.balls_input()
        input_data += self.oponents_input()
        return np.array(input_data)

    def player_input(self):
        player = [self.player.x,self.player.y,self.player.width,self.player.height,self.player.speed]
        for pos in self.player.bound:
            player += [pos[0],pos[1]]
        return player

    def balls_input(self):
        balls = []
        for ball in self.balls:
            ball_list = [ball.x,ball.y,math.cos(ball.direction),math.sin(ball.direction),ball.speed,ball.radius]
            balls +=ball_list
        balls += np.zeros(((self.max_balls - len(self.balls))*6)).tolist() # padding

        return balls



    def oponents_input(self):
        oponents = []

        for oponent in self.oponents:
            oponent_list = [oponent.x, oponent.y,oponent.width, oponent.height, oponent.speed]
            for pos in oponent.bound:
                oponent_list +=[pos[0], pos[1]]
            oponents += oponent_list
        oponents += np.zeros(((self.max_oponents - len(self.oponents))*13)).tolist()

        return oponents

    def move(self,dt):
        if self.call_count % self.how_often == 0 or self.call_count == 0:
            input_data = self.create_input()
            input_data = np.expand_dims(input_data, axis=0)
            actions = self.action(input_data)
            self.last_move = actions
            for oponent, action in zip(self.oponents,actions[0:len(self.oponents)]):
                oponent.move(float(action),dt)
        else:
            for oponent, action in zip(self.oponents,self.last_move[0:len(self.oponents)]):
                oponent.move(float(action),dt)
        self.call_count += 1

    def action(self, input_data):

        return self.model.predict(input_data, verbose=0)[0]

    def train(self, x_train, y_train, epochs=10, batch_size=8):
        #train
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, path):
        #save module
        self.model.save(path)