import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import math
import threading


class Bot():
    def __init__(self, player, oponents, balls,input_shape=(72,), model_path=None):
        # inputs: dt, how_often
        # player:[[x,y], [width, height], [speed], ]: 5
        # 5 * ball:[[x,y], [cos, sin], [speed, radius] distance]: 5 * 7
        # 6* oponent [[x,y], [width, height], [speed]] 6 * 5
        # -where the opondent it controls are att fist postition

        self.player = player
        self.oponents = oponents
        self.balls = balls

        self.how_often = 8
        self.call_count = 0
        #self.starting_distance = None
        self.past_states = []
        self.save_lenght = 1000

        self.max_balls = 5
        self.max_oponents = 6

        self.input_shape = input_shape

        self.model_pred = self.build_model(1/3) if model_path is None else tf.keras.models.load_model(model_path)
        self.model_train = self.build_model(1/10) if model_path is None else tf.keras.models.load_model(model_path)
        self.model_train.set_weights(self.model_pred.get_weights())
    def build_model(self,drop_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation = "relu",  kernel_regularizer= regularizers.l2(0.3)),
            tf.keras.layers.BatchNormalization(),  # Normalizes activations
            tf.keras.layers.Dropout(drop_rate/2),  # Prevents overfitting
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= regularizers.l2(0.3)),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer= regularizers.l2(0.3)),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(self.max_oponents, activation='tanh')  # Output range (-1 to 1) for movement for every oponent
        ])
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def create_input(self,dt):
        input_data = [dt,self.how_often]
        input_data += self.player_input()
        input_data += self.balls_input()
        input_data += self.oponents_input()
        return np.array(input_data)

    def player_input(self):
        player = [self.player.x,self.player.y,self.player.width,self.player.height,self.player.speed]
        return player

    def balls_input(self):
        balls = []
        for ball in self.balls:
            ball_list = [ball.x,ball.y,math.cos(ball.direction),math.sin(ball.direction),ball.speed,ball.radius,
                        ball.minimum_distance_to_paddle()]
            balls +=ball_list
        balls += np.zeros(((self.max_balls - len(self.balls))*7)).tolist() # padding

        return balls

    def oponents_input(self):
        oponents_list = []

        for oponent in self.oponents:
            oponents_list += [oponent.x, oponent.y,oponent.width, oponent.height, oponent.speed]
        oponents_list += np.zeros(((self.max_oponents - len(self.oponents))*5)).tolist()

        return oponents_list

    def normalize_weights(self,weights):
        weights = np.abs(weights)  # Absolutvärde
        weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)  # Hantera NaN eller inf



        max_val = np.max(weights)
        if max_val > 0:
            normalized = weights / (max_val/2 + 1e-8)   # Normalisera
            normalized = normalized**3
        else:
            normalized = np.zeros_like(weights)  # Fallback om max är 0

        return normalized + 1e-5

    def move(self,dt):
        if self.call_count % self.how_often == 0 or self.call_count == 0:
            input_data = self.create_input(dt)
            input_data = np.expand_dims(input_data, axis=0)
            actions = self.action(input_data)
            self.save_states(input_data,actions)
            for oponent, action in zip(self.oponents,actions[0:len(self.oponents)]):
                oponent.move(float(action),dt)
        else:
            for oponent, action in zip(self.oponents,self.past_states[-1][1][:len(self.oponents)]):
                oponent.move(float(action),dt)
        if self.call_count % self.save_lenght == 0 and not self.call_count == 0:
            self.active_training()
        self.call_count += 1

    def get_shortest_distance(self):
        return min([ball.minimum_distance_to_paddle() for ball in self.balls])

    def update_for_hit(self):
        for instiance in self.past_states[int(3/4*self.save_lenght if len(self.past_states) >3/4*self.save_lenght else
                                                                            len(self.past_states)):-2]:
            instiance[2] *= 2

    def save_states(self,input,output):
        distance = self.get_shortest_distance()
        self.past_states.append([input,output,distance])
        if len(self.past_states) > 1:
            self.past_states[-2][2] = np.exp(self.past_states[-2][2] - distance)
        if len(self.past_states) >= self.save_lenght:
            self.past_states.pop(0)

    def active_training(self):
        if len(self.past_states) >2:
            inputs, outputs, weights = map(np.array, zip(*self.past_states[:-2]))
            inputs = np.squeeze(inputs)

            weights = self.normalize_weights(weights)

            flip_mask = weights < 0.01
            outputs[flip_mask] *= -1.5
            outputs = np.clip(outputs, -1, 1)
            weights[flip_mask] = 0.2

            if not( len(inputs) == 0 or len(outputs) == 0 ):
                thread = threading.Thread(target=self.train, args=(inputs, outputs,weights))
                thread.start()

    def train(self, inputs, outputs, wheights,  epochs=6, batch_size=8):
        #train
        self.model_train.fit(inputs, outputs,sample_weight= wheights, epochs=epochs, verbose=2, batch_size = batch_size)
        self.model_pred.set_weights(self.model_train.get_weights())

    def action(self, input_data):
        return self.model_pred.predict(input_data, verbose=0)[0]

    def save_model(self, path):
        #save module
        self.model_train.save(path)