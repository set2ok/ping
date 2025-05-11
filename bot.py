import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import math
import threading

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

class Bot():
    """
    AI bot designed to control paddles in a game similar to ping pong. Uses neural network models to 
    predict and execute actions based on the current game state and past data. Trains adaptively 
    to improve performance over time.
    """

    def __init__(self, player: list, opponents: list, balls,input_shape=(53,), model_path="model_v5.keras", learning_rate = 0.001):
        """
        Initialize the bot with its attributes, models, and hyperparameters.
        can handel 6 opponents and 5 balls

        :param player: Details of the controlled paddle (position, size, speed, etc.).
        :param opponents: List of opponent paddles to manage interactions.
        :param balls: List of active balls in the game.
        :param input_shape: Shape of the input expected by the neural network.
        :param model_path: Path to a pre-trained model file.
        :param learning_rate: Learning rate for the model's optimizer.
        """
        
        
        
        # inputs: dt : 1
        # 5 * ball:[[x,y], [cos, sin], speed, distance, x col, y col,]: 5 * 8
        # 6* oponent [[x,y]] 6 * 2


        self.player = player
        self.opponents = opponents
        self.balls = balls

        self.how_often = 40
        self.call_count = 0
        #self.starting_distance = None
        self.past_states = []
        self.save_lenght = 2000
        self.is_training = False
        self.sequence_length = 5
        self.sequence_list = [np.zeros(input_shape[0]) for _ in range(self.sequence_length)]



        self.max_balls = 5
        self.max_opponents = 6

        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.epsilon = 0.2

        self.model_pred = self.build_model(1/50) if model_path is None else self.load_model(model_path)
        self.model_train = self.build_model(1/100) if model_path is None else self.load_model(model_path)
        self.model_train.set_weights(self.model_pred.get_weights())

    def build_model(self,drop_rate : float):
        """
        Build and compile a neural network model using TensorFlow/Keras.

        - The model consists of several LSTM and Dense layers with dropout for regularization.
        - have leaky ReLU activations to stop neurons from dying.
        - The output layer uses a 'tanh' activation function to ensure outputs are in the range (-1, 1).

        :param drop_rate: Float value representing the dropout rate for regularization.
        :return: A compiled Keras model for prediction or training.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.5)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.sequence_length,self.input_shape[0])),
            tf.keras.layers.LSTM(256, return_sequences=False),
            tf.keras.layers.Dense(512,  kernel_regularizer= regularizers.l2(0.008)),
            LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dropout(drop_rate/2),  # Prevents overfitting
            tf.keras.layers.Dense(256, kernel_regularizer= regularizers.l2(0.01)),
            LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(128, kernel_regularizer= regularizers.l2(0.01)),
            LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dropout(drop_rate),
            tf.keras.layers.Dense(self.max_opponents, activation='tanh')  # Output range (-1 to 1) for movement for every oponent
        ])
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def create_input(self,dt: float):
        """
        Create the input vector for the neural network by combining game state data.

        :param dt: Change in time since the last frame.
        :returns: A NumPy array containing the input data and a list of collision points for the balls.
        """
        input_data = [dt]
        input_balls, collision_points = self.balls_input()
        input_data += input_balls
        input_data += self.opponents_input()
        return np.array(input_data), collision_points

    def balls_input(self):
        """
        Encode the current state of all balls.
        Ball attributes: x, y, direction (cos, sin), speed, distance to paddle, collision point (x,y)
        padding with zeros if there are fewer than `max_balls`.
        :returns: balls input with padding and the colison points for the balls
        """
        balls = []
        colision_points = []
        for ball in self.balls:
            point = ball.next_colision()
            colision_points.append(point)
            ball_list = [ball.x,ball.y,math.cos(ball.direction),math.sin(ball.direction),ball.speed,
                        ball.minimum_distance_to_paddle(), point[0], point[1]]
            balls +=ball_list
        balls += np.zeros(((self.max_balls - len(self.balls))*8)).tolist() # padding

        return balls, colision_points

    def opponents_input(self):
        """
        Encode the current state of all opponent paddles.
        opponent attributes: x, y
        padding with zeros if there are fewer than `max_opponents`.
        :return: A padded list containing attributes of the opponents.
        """
        opponents_list = []

        for oponent in self.opponents:
            opponents_list += [oponent.x, oponent.y]
        opponents_list += np.zeros(((self.max_opponents - len(self.opponents))*2)).tolist()

        return opponents_list

    def create_input_sequence(self, current_input):
        """
        Update the input sequence used for prediction and return it.
        The sequence is a rolling window of the last `sequence_length` inputs.

        :param current_input: The latest input vector.
        :return: NumPy array containing the updated input sequence.
        """
        self.sequence_list.pop(0)
        self.sequence_list.append(current_input)
        return np.array(self.sequence_list)

    def move(self,dt: float):
        """
        Manage the movement of the bot's paddles by predicting or replaying actions.

        - At regular intervals (determined by `how_often`), it predicts actions for the paddles
            using the neural network. These actions are then applied to move the paddles.
        - Between prediction intervals, replay the last saved actions to reduce computational
            overhead.
        - If enough game states have been collected, perform active training to improve the model.

        :param dt: Delta time since the last frame update.
        """
        if self.call_count % self.how_often == 0 or self.call_count == 0:
            input_data, colision_points = self.create_input(dt)
            sequence = self.create_input_sequence(input_data)
            sequence = np.expand_dims(sequence, axis=0)
            actions = self.action(sequence)

            self.save_states(sequence,actions,colision_points) # save state output and colosion points

            for oponent, action in zip(self.opponents,actions[0:len(self.opponents)]):
                oponent.move(float(action),dt)
        else:
            for oponent, action in zip(self.opponents,self.past_states[-1][1][:len(self.opponents)]):
                oponent.move(float(action),dt)

        if (self.call_count % (self.save_lenght/2) == 0 and not self.call_count == 0
                and not len(self.past_states) <= self.save_lenght/2): #
            self.active_training()

        self.call_count += 1

    def distances_to_nearest_collision_point(self, colison_points):
        """
        Calculate the distance from each paddle to the next collision point of the balls.

        :param colison_points: List of collision points for each ball.
        :return: List of distances from each paddle to the nearest collision point.
        """

        distances = []
        for point in colison_points:
            distances_to_paddles = []
            for oponent in self.opponents:
                distances_to_paddles.append(math.sqrt((oponent.x - point[0])**2 + (oponent.y - point[1])**2))
            distances.append(min(distances_to_paddles))
        return distances


    def average_distance_between_paddles(self):
        """
        Calculate the average distance between each pair of paddles.

        :return: The average distance between all paddle pairs.
        """
        total_distance = 0
        pair_count = 0

        for i, paddle1 in enumerate(self.opponents):
            for j, paddle2 in enumerate(self.opponents):
                if i < j:  # Avoid duplicate pairs and self-comparison
                    dx = paddle1.x - paddle2.x
                    dy = paddle1.y - paddle2.y
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    total_distance += distance
                    pair_count += 1

        return total_distance / pair_count if pair_count > 0 else 0

    def adjust_weights_for_result(self, hit: bool, miss_distance = None):
        """
        Adjust the weights associated with stored states based on whether a paddle hit or missed the ball.

        - For successful hits, increase weights to reward good decisions.
        - For misses, penalize weights based on the miss distance to emphasize improvement.

        :param hit: Boolean indicating whether the paddle successfully hit the ball.
        :param miss_distance: Distance from the paddle when missing the ball.
        """
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
                scale = 1.0 - progress * 0.95  # 1.0 → 0.1
                penalty = scale / (penalty_strength + 1e-5)  # mindre avstånd → svagare straff
                self.past_states[i][2] *= penalty**(0.5)

    def save_states(self,input,output,colision_points):
        """
        Save a state consisting of input data, output actions, and a calculated distance metric
        for future training and weight adjustment.

        - Calculates a weighted distance for assessing the quality of the actions based on the difference in
            distance to balls and the average distance between etch opponent.
        - Removes the oldest data if the storage limit is exceeded.

        :param input: Input sequence data.
        :param output: Predicted actions corresponding to the input.
        """

        distances = self.distances_to_nearest_collision_point(colision_points)
        avg_distance = self.average_distance_between_paddles()


        self.past_states.append([input,output, ((distances,colision_points),avg_distance)])
        if len(self.past_states) > 1:
            old_points = self.distances_to_nearest_collision_point(self.past_states[-2][2][0][1])
            d_ball = (min([p for p in self.past_states[-2][2][0][0]])
                      - min([p for p in old_points]))
            d_avg_ball = (
                    sum([p for p in self.past_states[-2][2][0][0]])
                    - sum([p for p in old_points]))
            d_avg_paddle = self.past_states[-2][2][1] - avg_distance

            delta = (d_ball + d_avg_ball + d_avg_paddle)/3

            self.past_states[-2][2] = self.scaled_weight(delta)
        if len(self.past_states) >= self.save_lenght:
            self.past_states.pop(0)

    def scaled_weight(self,delta, base=1.3, max_scale=10.0):
        """
        Calculate a scaled weight based on the change in distance (delta).

        - A positive delta leads to an increase in weight (reward).
        - Enforces limits on scaling to prevent overly large adjustments.

        :param delta: Change in performance (distance difference).
        :param base: Base scaling factor.
        :param max_scale: Maximum allowable scale.
        :return: Scaled weight, clipped to the desired range.
        """
        scale = base ** delta  # t.ex. 1.2^delta
        return np.clip(scale, 1e-5, max_scale)

    def normalize_weights(self,weights):
        """
        Normalize the weights of the training data to ensure consistent training behavior.

        :param weights: List or array of weights to normalize.
        :return: Normalized weights (scaled between 0 and 1).
        """
        return weights / np.max(weights)

    def active_training(self, max = 20, min = 1e-5):
        """
        Perform active training on stored states to improve the model's predictions.

        - Selects the most relevant data points (states with the highest weights) and remove weights white no impact.
        - Prepares inputs, outputs, and weights for model training.
        - Normalizes weights when they reach the extremes.
        - Spawns a separate thread to avoid interrupting gameplay.

        :param max: Maximum allowable weight for scaling.
        :param min: Minimum allowable weight for scaling.
        """
        if len(self.past_states) >2:
            data = self.past_states[:int(self.save_lenght/2)]
            data.sort(key=lambda x: x[2], reverse=True)
            data = data[:int(len(data)/2)]
            inputs, outputs, weights = map(np.array, zip(*data))
            inputs = np.squeeze(inputs)

            avg_wheight = np.average(weights)
            if avg_wheight >= max:
                weights = self.normalize_weights(weights) * max
            elif avg_wheight <= min:
                weights = self.normalize_weights(weights)

            mask = weights > 1e-2
            weights = weights[mask]
            inputs = inputs[mask]
            outputs = outputs[mask]

            weights = np.clip(weights,min,max)
            outputs = np.clip(outputs,-0.98,0.98)


            if not( len(inputs) == 0 or len(outputs) == 0 ):
                if not self.is_training:
                    self.is_training = True
                    thread = threading.Thread(target=self.train, args=(inputs, outputs,weights))
                    thread.start()

    def train(self, inputs, outputs, wheights,  epochs=3, batch_size=8):
        """
        Train the bot's model using the provided training data.

        - Trains the `model_train` neural network using stored game states as inputs and
            corresponding actions as outputs with weighted importance given to higher-priority cases.
        - After training, updates the `model_pred` weights to align with the improved `model_train`.
        - Outputs metrics (mean, standard deviation, etc.) for debugging and analysis.

        :param inputs: Array of input sequences representing game states.
        :param outputs: Array of actions corresponding to the inputs.
        :param weights: Array of weights assigning importance to each state.
        :param epochs: Number of passes over the training data.
        :param batch_size: Number of samples used per training step.
        """

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
        """
        Predict actions for the bot's paddles based on the current game state.

        - Chooses actions either from uniform random exploration (epsilon-greedy policy) or
            from the predictions of the neural network with added noise for diversity.
        - Gradually reduces the exploration factor (`epsilon`) over time to favor learned policies.

        :param input_data: Input sequence (game state) used for prediction.
        :return: NumPy array of predicted actions for each opponent.
        """

        self.epsilon = max(0.01, self.epsilon * 0.9999)
        if np.random.rand() < self.epsilon:

            return np.random.uniform(-1.0, 1.0, size=self.max_opponents)

        else:
            pred = self.model_pred.predict(input_data, verbose=0)[0]
            confidence = np.std(pred)  # hög std = osäkrare modell
            noise_scale = 0.05 + 0.3 * confidence
            noise = np.random.normal(0, noise_scale, size=pred.shape)
            return np.clip(pred + noise, -1.0, 1.0)

    def load_model(self, path : str):
        """
        Load a pre-trained neural network model from a specified path.

        - Compiles the loaded model with the bot's learning configurations.

        :param path: Path to the saved `.keras` model file.
        :return: The loaded and compiled Keras model.
        """

        model =  tf.keras.models.load_model(path)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.5)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())
        return model

    def save_model(self, path = "model_v5.keras"):
        """
        Save the bot's training model to a specified path.

        - Ensures that the training model (`model_train`) is saved for later use.

        :param path: Path to save the `.keras` model file.
        """
        self.model_train.save(path)