import tensorflow as tf
from tensorflow.keras import layers, regularizers
import numpy as np
import math
import threading


class Bot():
    def __init__(self, player: list, oponents: list, balls,input_shape=(72,), model_path=None):
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
        self.save_lenght = 3000

        self.max_balls = 5
        self.max_oponents = 6

        self.input_shape = input_shape

        self.model_pred = self.build_model(1/6) if model_path is None else tf.keras.models.load_model(model_path)
        self.model_train = self.build_model(1/20) if model_path is None else tf.keras.models.load_model(model_path)
        self.model_train.set_weights(self.model_pred.get_weights())
    def build_model(self,drop_rate : float):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.Dense(256, activation = "relu",  kernel_regularizer= regularizers.l2(0.1)),
            tf.keras.layers.Dropout(drop_rate/2),  # Prevents overfitting
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer= regularizers.l2(0.1)),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer= regularizers.l2(0.1)),
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

    def move(self,dt: float):
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

    def adjust_weights_for_result(self, hit: bool):

        n = len(self.past_states)
        if n < 2:
            return

        start = int(0.25 * n)
        if hit:
            for i in range(start, n):
                progress = (i - start) / (n - start)  # 0 → 1
                scale = 1.0 + progress * 1.5  # 1.0 → 2.5
                self.past_states[i][2] *= scale
        else:
            miss_distance = self.get_shortest_distance()
            penalty_strength = np.clip(miss_distance, 0.1, 3.0)  # undvik överdrift

            for i in range(start, n):
                progress = (i - start) / (n - start)
                scale = 1.0 - progress * 0.9  # 1.0 → 0.1
                penalty = scale / (penalty_strength + 1e-5)  # mindre avstånd → svagare straff
                self.past_states[i][2] *= penalty

    def save_states(self,input,output):
        distance = self.get_shortest_distance()
        self.past_states.append([input,output,distance])
        if len(self.past_states) > 1:
            delta = self.past_states[-2][2] - distance
            self.past_states[-2][2] = self.scaled_weight(delta)
        if len(self.past_states) >= self.save_lenght:
            self.past_states.pop(0)

    def scaled_weight(self,delta, base=1.2, max_scale=5.0):
        # Begränsa delta så inte vikterna exploderar eller kollapsar
        delta = np.clip(delta, -3, 3)
        scale = base ** delta  # t.ex. 1.2^delta
        return np.clip(scale, 0.01, max_scale)

    def active_training(self):
        if len(self.past_states) >2:
            inputs, outputs, weights = map(np.array, zip(*self.past_states[:-2]))
            inputs = np.squeeze(inputs)


            if not( len(inputs) == 0 or len(outputs) == 0 ):
                thread = threading.Thread(target=self.train, args=(inputs, outputs,weights))
                thread.start()

    def train(self, inputs, outputs, wheights,  epochs=6, batch_size=8):

        print("Output stats before training:")
        print("Mean:", np.mean(outputs, axis=0))
        print("Std Dev:", np.std(outputs, axis=0))
        print("Weights stats:")
        print("Min:", np.min(wheights), "Max:", np.max(wheights))

        #train
        self.model_train.fit(inputs, outputs,sample_weight= wheights, epochs=epochs, verbose=2)
        self.model_pred.set_weights(self.model_train.get_weights())

    def action(self, input_data):
        pred = self.model_pred.predict(input_data, verbose=0)[0]
        noise = np.random.normal(0, 0.2, size=pred.shape)
        return np.clip(pred + noise, -1.0, 1.0)

    def save_model(self, path):
        #save module
        self.model_train.save(path)