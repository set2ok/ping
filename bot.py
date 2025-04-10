import tensorflow as tf
import numpy as np
import math


class Bot():
    def __init__(self, player, oponents, balls,powerups,input_shape=(13,7,2), model_path=None):
        # inputs: player:[[x,y], [width, height], [speed,0], 4*[bound]],
        # 4 * ball:[[x,y], [cos, sin], [speed, radius]],  3* powerup: [[x,y], [width, height],[power, duration]]
        # 5* oponent [[x,y], [width, height], [speed,0], 4*[bound]]
        # -where the opondent it controls are att fist postition

        self.input_shape = input_shape
        self.model = self.build_model() if model_path is None else tf.keras.models.load_model(model_path)

        self.player = player
        self.oponents = oponents
        self.balls = balls
        self.powerups = powerups

        self.max_balls = 4
        self.max_oponents = 5
        self.max_powerups = 3
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),  # Normalizes activations
            tf.keras.layers.Dropout(0.01),  # Prevents overfitting
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='tanh')  # Output range (-1 to 1) for movement
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_base_input(self):
        input_base_data = []
        input_base_data.append(self.player_input())
        input_base_data += self.balls_input()
        input_base_data += self.powerups_inputs()
        return input_base_data

    def player_input(self):
        player = []
        player.append([self.player.x,self.player.y])
        player.append([self.player.width,self.player.height])
        player.append([self.player.speed,0])
        for pos in self.player.bound:
            player.append([pos[0],pos[1]])
        return player

    def balls_input(self):
        balls = []
        for ball in self.balls:
            ball_list = []
            ball_list.append([ball.x,ball.y])
            ball_list.append([math.cos(ball.direction),math.sin(ball.direction)])
            ball_list.append([ball.speed,ball.radius])
            ball_list += np.zeros(((self.input_shape[1] - 3),self.input_shape[2])) # padding
            balls.append(ball_list)
        balls += np.zeros(((self.max_balls - len(self.balls)),(self.input_shape[1]),self.input_shape[2])) # padding

        return balls

    def powerups_inputs(self):
        # dont have powerups for now
        powerups = np.zeros((self.max_powerups,self.input_shape[1],self.input_shape[2]))
        return powerups


    def oponents_input(self,oponent):
        oponents = []
        active_oponent = []
        active_oponent.append([oponent.x,oponent.y])
        active_oponent.append([oponent.width,oponent.height])
        active_oponent.append([oponent.speed,0])
        for pos in oponent.bound:
            active_oponent.append([pos[0],pos[1]])
        oponents.append(active_oponent)

        for paddle in self.oponents:
            if not paddle == oponent:
                paddle_list = []
                paddle_list.append([paddle.x, paddle.y])
                paddle_list.append([paddle.width, paddle.height])
                paddle_list.append([paddle.speed, 0])
                for pos in paddle.bound:
                    paddle_list.append([pos[0], pos[1]])
                oponents.append(paddle_list)
        oponents += np.zeros(((self.max_oponents - len(self.oponents)),self.input_shape[1],self.input_shape[2])) #padding

    def move(self,dt):
        base_input = self.create_base_input()
        for oponent in self.oponents:
            input_data = base_input + self.oponents_input(oponent)
            action = self.action(input_data)
            oponent.move(action,dt)

    def action(self, input_data):
        #
        movement = self.model.predict(input_data, verbose=0)[0][0]
        return movement  # Can be used to move up/down

    def train(self, x_train, y_train, epochs=10, batch_size=8):
        #train
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def save_model(self, path):
        #save module
        self.model.save(path)