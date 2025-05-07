import tensorflow as tf
from setuptools.dist import sequence
from tensorflow.keras import layers, regularizers
import numpy as np
import math
import threading

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

class Bot():
    def __init__(self, player: list, oponents: list, balls,input_shape=(72,), model_path=None, learning_rate = 0.01):
        # inputs: dt, how_often : 2
        # player:[[x,y], [width, height], [speed], ]: 5
        # 5 * ball:[[x,y], [cos, sin], [speed, radius] distance]: 5 * 7
        # 6* oponent [[x,y], [width, height], [speed]] 6 * 5
        # -where the opondent it controls are att fist postitionno


        self.player = player
        self.oponents = oponents
        self.balls = balls

        self.how_often = 20
        self.call_count = 0
        #self.starting_distance = None
        self.past_states = []
        self.save_lenght = 4000
        self.is_training = False
        self.sequence_length = 5



        self.max_balls = 5
        self.max_oponents = 6

        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.epsilon = 0.1

        self.model_pred = self.build_model(1/6) if model_path is None else self.load_model(model_path)
        self.model_train = self.build_model(1/20) if model_path is None else self.load_model(model_path)
        self.model_train.set_weights(self.model_pred.get_weights())

    def build_model(self,drop_rate : float):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.5)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.sequence_length,self.input_shape[0])),
            tf.keras.layers.LSTM(128, return_sequences=False),
            tf.keras.layers.Dense(128, activation = "relu",  kernel_regularizer= regularizers.l2(0.01)),
            tf.keras.layers.Dropout(drop_rate/2),  # Prevents overfitting
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= regularizers.l2(0.01)),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= regularizers.l2(0.01)),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(self.max_oponents, activation='tanh')  # Output range (-1 to 1) for movement for every oponent
        ])
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def create_input(self,dt: float):
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

    def create_input_sequence(self, current_input):


        sequence = [state[0][:] for state in self.past_states[-(self.sequence_length - 1):]]
        sequence.append(current_input)

        if len(sequence) < self.sequence_length: # pad with zeros
            padding = [np.zeros_like(self.)] * (self.sequence_length - len(sequence))
            sequence = padding + sequence


        return np.array(sequence)

    def move(self,dt: float):
        if self.call_count % self.how_often == 0 or self.call_count == 0:
            input_data = self.create_input(dt)
            sequence = self.create_input_sequence(input_data)
            sequence = np.expand_dims(sequence, axis=0)
            actions = self.action(sequence)
            self.save_states(sequence,actions)
            for oponent, action in zip(self.oponents,actions[0:len(self.oponents)]):
                oponent.move(float(action),dt)
        else:
            for oponent, action in zip(self.oponents,self.past_states[-1][1][:len(self.oponents)]):
                oponent.move(float(action),dt)
        if self.call_count % (self.save_lenght/2) == 0 and not self.call_count == 0 and not len(self.past_states) <= self.save_lenght/2:
            self.active_training()
        self.call_count += 1

    def get_distance_to_balls(self):
        return [ball.minimum_distance_to_paddle() for ball in self.balls]

    def adjust_weights_for_result(self, hit: bool, miss_distance = None):

        n = len(self.past_states) - 1
        if n < 2:
            return

        start = int(0.25 * n)
        if hit:
            for i in range(start, n):
                progress = (i - start) / (n - start)  # 0 → 1
                scale = 1.0 + progress * 2
                self.past_states[i][2] *= scale
        else:
            penalty_strength = np.clip(miss_distance, 0.01, 6.0)  # undvik överdrift

            for i in range(start, n):
                progress = (i - start) / (n - start)
                scale = 1.0 - progress * 0.9  # 1.0 → 0.1
                penalty = scale / (penalty_strength + 1e-5)  # mindre avstånd → svagare straff
                self.past_states[i][2] *= penalty**(0.5)

    def save_states(self,input,output):
        distances = self.get_distance_to_balls()
        distance = min(distances)
        if len(self.balls) > 1:
            distances.remove(distance)
            distance = (distance + min(distances))/2

        self.past_states.append([input,output, distance])
        if len(self.past_states) > 1:
            delta = self.past_states[-2][2] - distance
            self.past_states[-2][2] = self.scaled_weight(delta)
        if len(self.past_states) >= self.save_lenght:
            self.past_states.pop(0)

    def scaled_weight(self,delta, base=1.2, max_scale=6.0):
        scale = base ** delta  # t.ex. 1.2^delta
        return np.clip(scale, 1e-5, max_scale)

    def normalize_weights(self,weights):
        return weights / np.max(weights)

    def active_training(self, max = 20, min = 1e-5):
        if len(self.past_states) >2:
            data = self.past_states[:int(self.save_lenght/2)]
            data.sort(key=lambda x: x[2], reverse=True)
            data = data[:int(len(data)/4)]
            inputs, outputs, weights = map(np.array, zip(*data))
            inputs = np.squeeze(inputs)

            avg_wheight = np.average(weights)
            if avg_wheight >= max:
                weights = self.normalize_weights(weights) * max
            elif avg_wheight <= min:
                weights = self.normalize_weights(weights)

            weights = np.clip(weights,min,max)
            outputs = np.clip(outputs,-0.98,0.98)


            if not( len(inputs) == 0 or len(outputs) == 0 ):
                if not self.is_training:
                    self.is_training = True
                    thread = threading.Thread(target=self.train, args=(inputs, outputs,weights))
                    thread.start()

    def train(self, inputs, outputs, wheights,  epochs=5, batch_size=8):

        print("Output stats before training:")
        print("Mean:", np.mean(outputs, axis=0))
        print("Std Dev:", np.std(outputs, axis=0))
        print("Weights stats:")
        print("Min:", np.min(wheights), "Max:", np.max(wheights))

        #train
        self.model_train.fit(inputs, outputs,sample_weight= wheights, epochs=epochs, verbose=2)
        self.model_pred.set_weights(self.model_train.get_weights())

        self.is_training = False

    def action(self, input_data):
        self.epsilon = max(0.01, self.epsilon * 0.9999)
        if np.random.rand() < self.epsilon:

            return np.random.uniform(-1.0, 1.0, size=self.max_oponents)

        else:
            pred = self.model_pred.predict(input_data, verbose=0)[0]
            confidence = np.std(pred)  # hög std = osäkrare modell
            noise_scale = 0.05 + 0.3 * confidence
            noise = np.random.normal(0, noise_scale, size=pred.shape)
            return np.clip(pred + noise, -1.0, 1.0)

    def load_model(self, path : str):
        model =  tf.keras.models.load_model(path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def save_model(self, path = "model_v1.keras"):
        #save module
        self.model_train.save(path)