import gym
import gym_minecraft
import numpy as np
import tensorflow.compat.v1 as tf
import math
import random
import time
import datetime
import pickle
from enum import Enum
import matplotlib.pyplot as plt
from lxml import etree
import logging
import mcenv
import sys
import csv
sys.path.append("..")


graph1 = tf.Graph()
frame_size = [40,40]
num_input = frame_size[0] * frame_size[1]


env = gym.make('MinecraftEnv-v0')
env.init(start_minecraft = False,
         videoResolution = [frame_size[0], frame_size[1]],
         allowDiscreteMovement = ["move", "jumpsouth"],
         step_sleep = 0,
         skip_steps = 0) #Movements modified to a faster convergence

class LSTM():
    def __init__(self, rnn_cell, scope):
        self.x = tf.placeholder("float", [None, num_input])
        # Length of the frames' sequence
        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])

        # Reshape the flatten data
        self.input_layer = tf.reshape(self.x, [-1, frame_size[1], frame_size[0], 1])

        # Convolutional Layer 1
        self.conv1 = tf.layers.conv2d(
            inputs=self.input_layer,
            filters=32,
            kernel_size=[6, 6],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)
        # Output size = 28

        # Convolutional Layer 2
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=64,
            kernel_size=[6, 6],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)
        # Output size = 12

        # Convolutional Layer 3
        self.conv3 = tf.layers.conv2d(
            inputs=self.conv2,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding="valid",
            activation=tf.nn.relu)
        # Output size = 5

        # Flatten the data before passing it through the recurrent layer
        self.dims = self.conv3.get_shape().as_list()
        self.final_dimension = self.dims[1] * self.dims[2] * self.dims[3]
        self.conv3_flat = tf.reshape(self.conv3, [-1, self.final_dimension])
        self.rnn_input = tf.reshape(self.conv3_flat, [self.batch_size, self.train_length, self.final_dimension])

        # Initialize the LSTM state
        self.lstm_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.rnn_input, cell=rnn_cell, dtype=tf.float32, initial_state=self.lstm_state_in, scope=scope + "_rnn")
        self.rnn = tf.reshape(self.rnn, shape=[-1, num_nodes])

        # Feed Forward
        self.dense = tf.layers.dense(inputs=self.rnn, units=512, activation=tf.nn.relu)

        self.Qout = tf.layers.dense(inputs=self.dense, units=num_classes)

        # Indexes of the actions the network shall take
        self.prediction = tf.argmax(self.Qout, 1)

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # Multiply our actions values by a OneHotEncoding to only take the chosen ones.
        self.actions_onehot = tf.one_hot(self.actions, num_classes, dtype=tf.float32)
        # So that Q's going to be the Q-values chosen by the Target network
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        # NextQ corresponds to the Q estimated by the Bellman Equation
        self.nextQ = tf.placeholder(shape=[None], dtype=tf.float32)

        # The loss value coresponds to the difference between the two different Q-values estimated
        self.loss = tf.reduce_mean(tf.square(self.nextQ - self.Q))

        # Let's print the important informations
        self.merged = tf.summary.merge([tf.summary.histogram("nextQ", self.nextQ),
                                        tf.summary.histogram("Q", self.Q),
                                        tf.summary.scalar("Loss", self.loss)])

        self.learningRate = learningRate
        # We would prefer the Adam Optimizer
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.updateModel = self.trainer.minimize(self.loss)
#
# LSTM = LSTM()
# trainer = LSTM.trainer

##########################################################################################################
class TensorBoardInfosLogger():
    def __init__(self):
        self.percent_win = tf.placeholder(dtype=tf.float32)
        self.mean_j_by_win = tf.placeholder(dtype=tf.float32)
        self.mean_rewards_sum = tf.placeholder(dtype=tf.float32)
        self.merged = tf.summary.merge([tf.summary.scalar("Percent_of_win_on_last_50_episodes", self.percent_win),
                                        tf.summary.scalar("Number_of_steps_by_win_on_last_50_episodes", self.mean_j_by_win),
                                        tf.summary.scalar("Mean_of_sum_of_rewards_on_last_50_episodes", self.mean_rewards_sum), ])

def processState(state):
    gray_state = np.dot(state[...,:3], [0.299, 0.587, 0.114]) #Downscale input to greyscale
    return np.reshape(gray_state, num_input)/255.0 #Normalize pixels

def reverse_processState(state):
    return np.reshape(state, (frame_size[0], frame_size[1]))*255.0

def get_stacked_states(episode_frames, trace_length): #Fills the stacked frames with images full of zero if the sequence is too short
    if len(episode_frames) < trace_length:
        nb_missing_states = trace_length - len(episode_frames)
        zeros_padded_states = [np.zeros(num_input) for _ in range(nb_missing_states)]
        zeros_padded_states.extend(episode_frames)
        return np.reshape(np.array(zeros_padded_states), num_input*trace_length)
    else:
        return np.reshape(np.array(episode_frames[-trace_length:]), num_input*trace_length)

##########################################################################################################
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]): #Get the weights of the original network
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value()))) #Update the Target Network weights
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

##########################################################################################################
class experience_buffer():
    def __init__(self, buffer_size=200000):  # Stores steps
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])


class recurrent_experience_buffer():
    def __init__(self, buffer_size=5000):  # Stores episodes
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.buffer if len(episode) + 1 > trace_length]
        # print("=========tmp, batch", tmp_buffer, batch_size)
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 5])

    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])


class stacked_experience_buffer():
    def __init__(self, buffer_size=5000):  # Stores episodes
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        tmp_buffer = [episode for episode in self.buffer if len(episode) + 1 > trace_length]
        sampled_episodes = random.sample(tmp_buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            stacked_states = []
            stacked_next_states = []
            for trace in episode[point:point + trace_length]:
                stacked_states.extend(trace[0])
                stacked_next_states.extend(trace[3])
            trace_to_return = episode[point + trace_length - 1].copy()
            trace_to_return[0] = stacked_states
            trace_to_return[3] = stacked_next_states
            sampledTraces.append(trace_to_return)
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size, 5])

    def get(self):
        return np.reshape(np.array(self.buffer), [len(self.buffer), 5])

##########################################################################################################

#Training

##########################################################################################################
NetType = LSTM
learningRate = 0.005
num_nodes = 256
num_classes = len(env.action_names[0])
batch_size =  18 #How many steps to use for each training step.
trace_length = 3 #How long each experience trace will be when training

myBuffer = recurrent_experience_buffer()
batch_size = int(batch_size / trace_length)


# update_freq = 4 #How often to perform a training step.
# num_episodes = 100000 #How many episodes of game environment to train network with
# total_steps = 0
# rList = [] #List of our rewards gained by game
# jList = [] #Number of moves realised by game
# j_by_loss = [] #Number of moves before resulting with a death of the agent
# j_by_win = [] #Number of moves before resulting with a win of the agent
# j_by_nothing = [] #This list's going to be used to count how many times the agent moves until the limit of moves is reached
# y = .95 #Discount factor on the target Q-values
#
# ## Exploration Settings
#
# pre_train_steps = 1000 #How many episodes of random actions before training begins.
# startE = 1 #Starting chance of random action
# endE = 0.1 #Final chance of random action
# annealing_steps = 200000 #How many epsiodes of training to reduce startE towards endE.
# e = startE
# stepDrop = (startE - endE) / annealing_steps
# nb_win = 0
# nb_win_tb = 0
# nb_nothing = 0
# nb_loss = 0
# tau = 0.001
# load_model = False

FILE_NAME = "results_lstm6.csv"
f = open(FILE_NAME, mode="w")
quick_writer = csv.writer(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

quick_writer.writerow([learningRate])
quick_writer.writerow([num_nodes])
quick_writer.writerow([batch_size])


is_debug = False
if is_debug:
    write_path = 'train/test'

def print_debug_states(tf_session, QNet, raw_input, trace_length):
    tmp = tf_session.run(QNet.input_layer, feed_dict={QNet.x:[raw_input]})
    for depth in range(tmp.shape[3]):
        print("## Input image n°" + str(depth) + " ##")
        plt.imshow(tmp[0, :, :, depth], cmap=plt.get_cmap('gray'))
        plt.show()

## Training

def train():
    update_freq = 4  # How often to perform a training step.
    num_episodes = 1000  # How many episodes of game environment to train network with
    total_steps = 0
    rList = []  # List of our rewards gained by game
    jList = []  # Number of moves realised by game
    j_by_loss = []  # Number of moves before resulting with a death of the agent
    j_by_win = []  # Number of moves before resulting with a win of the agent
    j_by_nothing = []  # This list's going to be used to count how many times the agent moves until the limit of moves is reached
    y = .95  # Discount factor on the target Q-values

    ## Exploration Settings

    pre_train_steps = 200  # How many episodes of random actions before training begins.
    startE = 1  # Starting chance of random action
    endE = 0.1  # Final chance of random action
    annealing_steps = 200000  # How many epsiodes of training to reduce startE towards endE.
    e = startE
    stepDrop = (startE - endE) / annealing_steps
    nb_win = 0
    nb_win_tb = 0
    nb_nothing = 0
    nb_loss = 0
    tau = 0.001
    load_model = False

    date = str(time.time()).replace(".", "")
    net = str(NetType).split(".")[1]
    bs = "BatchSize-" + str(batch_size)
    strlr = "lr-" + str(learningRate)
    rand_step = "RandStep-" + str(pre_train_steps)
    nb_to_reduce_e = "ReducE-" + str(annealing_steps)
    write_path = "train/" + net + "_" + bs + "_" + strlr + "_" + rand_step + "_" + nb_to_reduce_e + "_" + date[-5:]

    tf.reset_default_graph()
    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_nodes, state_is_tuple=True)
        cellT = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_nodes, state_is_tuple=True)

        mainQN = NetType(cell, 'main')
        targetQN = NetType(cellT, 'target')

        trainables = tf.trainable_variables()
        targetOps = updateTargetGraph(trainables, tau)

        # Save the network
        saver = tf.train.Saver()
        path_to_save = "saves/" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + "/"

        init = tf.global_variables_initializer()
        sess.run(init)

        tb_infos_logger = TensorBoardInfosLogger()
        writer = tf.summary.FileWriter(write_path)

        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path_to_save)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            lstm_state = (
            np.zeros([1, num_nodes]), np.zeros([1, num_nodes]))  # Reset the recurrent layer's hidden state

            episodeBuffer = experience_buffer()

            s = env.reset()
            s = processState(s)
            d = False
            j = 0
            episode_frames = []
            episode_frames.append(s)
            episode_qvalues = []
            episode_rewards = []

            ZPos = []
            XPos = []
            Yaw = []
            moves = []

            while not d:
                j += 1

                ### Epsilon Greedy ###
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop

                # Make full exploration before the number of pre-train episodes then play with an e chance of random action during the training (e-greedy)
                if (np.random.rand(1) < e or total_steps < pre_train_steps):
                    lstm_state1 = sess.run(mainQN.rnn_state,
                                           feed_dict={mainQN.x: [s], mainQN.train_length: 1,
                                                      mainQN.lstm_state_in: lstm_state, mainQN.batch_size: 1})
                    index_action_predicted = env.action_space.sample()
                    episode_qvalues.append(
                        [1 if i == index_action_predicted else 0 for i in range(len(env.action_names[0]))])
                else:
                    if is_debug:
                        print_debug_states(sess, mainQN, s, trace_length)

                    prediction, qvalues, lstm_state1 = sess.run([mainQN.prediction, mainQN.Qout, mainQN.rnn_state], \
                                                                feed_dict={mainQN.x: [s], mainQN.train_length: 1, \
                                                                           mainQN.lstm_state_in: lstm_state,
                                                                           mainQN.batch_size: 1})
                    index_action_predicted = prediction[0]
                    episode_qvalues.append(qvalues[0])

                # Get new state and reward from environment
                s1_raw, r, d, info = env.step(index_action_predicted)

                if info["observation"]:
                    ZPos.append(info['observation']['ZPos'])
                    XPos.append(info['observation']['XPos'])
                    Yaw.append(info['observation']['Yaw'])
                s1 = processState(s1_raw)
                moves.append(index_action_predicted)
                episodeBuffer.add(np.reshape(np.array([s, index_action_predicted, r, s1, d]), [1, 5]))
                episode_frames.append(s1)

                if total_steps > pre_train_steps:
                    if total_steps % (update_freq) == 0:

                        updateTarget(targetOps, sess)  # Update Target Network

                        lstm_state_train = (np.zeros([batch_size, num_nodes]),
                                            np.zeros([batch_size, num_nodes]))

                        # print("----batch, trace", batch_size, trace_length)
                        trainBatch = myBuffer.sample(batch_size, trace_length)

                        if is_debug:
                            print_debug_states(sess, mainQN, trainBatch[0, 0], trace_length)

                        # Estimate the action to choose by our first network
                        actionChosen = sess.run(mainQN.prediction,
                                                feed_dict={mainQN.x: np.vstack(trainBatch[:, 3]),
                                                           mainQN.train_length: trace_length,
                                                           mainQN.lstm_state_in: lstm_state_train,
                                                           mainQN.batch_size: batch_size})
                        # Estimate all the Q-values by our second network --> Double
                        allQValues = sess.run(targetQN.Qout,
                                              feed_dict={targetQN.x: np.vstack(trainBatch[:, 3]),
                                                         targetQN.train_length: trace_length,
                                                         targetQN.lstm_state_in: lstm_state_train,
                                                         targetQN.batch_size: batch_size})

                        # Train our network using target and predicted Q-values
                        end_multiplier = -(trainBatch[:, 4] - 1)
                        maxQ = allQValues[range(batch_size * trace_length), actionChosen]
                        # Bellman Equation
                        targetQ = trainBatch[:, 2] + (y * maxQ * end_multiplier)

                        _, summaryPlot = sess.run([mainQN.updateModel, mainQN.merged],
                                                  feed_dict={mainQN.x: np.vstack(trainBatch[:, 0]),
                                                             mainQN.nextQ: targetQ,
                                                             mainQN.actions: trainBatch[:, 1],
                                                             mainQN.train_length: trace_length,
                                                             mainQN.lstm_state_in: lstm_state_train,
                                                             mainQN.batch_size: batch_size})

                        writer.add_summary(summaryPlot, total_steps)

                episode_rewards.append(r)
                if (s == s1).all():
                    print("State error : State did not changed though the action was " + env.action_names[0][
                        index_action_predicted])

                s = s1
                total_steps += 1
                lstm_state = lstm_state1

                if d == True:
                    if r == 0 or r == 10 or r == -10:
                        print("Unrecognized reward Error : " + str(r))
                        j_by_loss.append(j)
                    elif r > 0:
                        j_by_win.append(j)
                    elif r < 0:
                        j_by_nothing.append(j)
                    break

            myBuffer.add(episodeBuffer.buffer)
            jList.append(j)
            rList.append(sum(episode_rewards))
            rewards = np.array(rList)
            print("Episode: ", i, " |Cumulative Rewards: ", sum(episode_rewards))
            quick_writer.writerow([sum(episode_rewards)])

            if i % (50) == 0:
                nb_of_win_on_last_50 = (len(j_by_win) - nb_win_tb)
                win_perc = nb_of_win_on_last_50 / 50 * 100
                mean_j_by_win = np.mean(j_by_win[-nb_of_win_on_last_50:])
                mean_rewards_sum = np.mean(rList[-50:])
                summaryPlot = sess.run(tb_infos_logger.merged,
                                       feed_dict={tb_infos_logger.percent_win: win_perc, \
                                                  tb_infos_logger.mean_j_by_win: mean_j_by_win, \
                                                  tb_infos_logger.mean_rewards_sum: mean_rewards_sum})
                writer.add_summary(summaryPlot, i)
                nb_win_tb = len(j_by_win)

            if i % (500) == 0:
                print("#######################################")
                print("% Win : " + str((len(j_by_win) - nb_win) / 5) + "%")
                print("% Nothing : " + str((len(j_by_nothing) - nb_nothing) / 5) + "%")
                print("% Loss : " + str((len(j_by_loss) - nb_loss) / 5) + "%")

                print("Nb J before win: " + str(np.mean(j_by_win[-(len(j_by_win) - nb_win):])))
                print("Nb J before die: " + str(np.mean(j_by_loss[-(len(j_by_loss) - nb_loss):])))

                print("Total Steps: " + str(total_steps))
                print("I: " + str(i))
                print("Epsilon: ", str(e))

                nb_win = len(j_by_win)
                nb_nothing = len(j_by_nothing)
                nb_loss = len(j_by_loss)

                # print("#### LAST 5 MOVES of LAST EPISODE  ####")
                # last_episode_moves = episodeBuffer.get()
                # starting_point = j - 5 if j >= 5 else 0
                # for z in range(starting_point, j):
                #     print("-----------------------")
                #     plt.imshow(reverse_processState(episode_frames[z]), cmap=plt.get_cmap("gray"))
                #     plt.show()
                #
                #     print("- Buffer Move " + str(z) + " : " + env.action_names[0][last_episode_moves[z, 1]])
                #     print("- Move Array " + str(z) + " : " + env.action_names[0][moves[z]])
                #     if z != j - 1:
                #         print("ZPos : " + str(ZPos[z]))
                #         print("XPos : " + str(XPos[z]))
                #         print("Yaw : " + str(Yaw[z]))
                #     figure = plt.figure()
                #     axes = figure.add_subplot(2, 1, 1)
                #     axes.matshow([episode_qvalues[z]])
                #     axes.set_xticks(range(len(env.action_names[0])))
                #     actions_names = ["Straight", "Back", "Right", "Left"]
                #     axes.set_xticklabels(actions_names)
                #     plt.show()
                #
                #     print("         " + "          ".join(str(qval) for qval in episode_qvalues[z]))
                #     print("Obtained reward : " + str(episode_rewards[z]))

            if i % (5000) == 0 and i != 0:
                # Save all the other important values
                saver.save(sess, path_to_save + str(i) + '.ckpt')
                with open(path_to_save + str(i) + ".pickle", 'wb') as file:
                    dictionnary = {
                        "epsilon": e,
                        "Total_steps": total_steps,
                        "Buffer": myBuffer,
                        "rList": rList,
                        "Num Episodes": i,
                        "jList": jList}

                    pickle.dump(dictionnary, file, protocol=pickle.HIGHEST_PROTOCOL)

        # saver.save(sess, path_to_save + str(i) + '.ckpt')


try:
    train()
    f.close()
except Exception as e:
    f.close()
    print(e)
except BaseException as be:
    f.close()
    print(be)