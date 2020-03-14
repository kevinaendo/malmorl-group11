import csv
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = "results_lstm5.csv"


def parse_file(f): 
	csv_reader = csv.reader(f, delimiter=",")

	learning_rate = -1
	num_nodes = -1
	batch_size = -1
	rewards = []

	for index, value in enumerate(csv_reader): 
		# Get learning rate
		if index == 0: 
			learning_rate = value[0]

		# Get num_nodes
		elif index == 1:
			num_nodes = value[0]
		elif index == 2:
			batch_size = value[0]

		# Get rewards list
		else: 
			rewards.append(value[0])

	return (learning_rate, num_nodes, batch_size, rewards)


def graph(data):

	WINDOW_LEN = 10
	learning_rate = data[0]
	datas = np.asarray(data[3], dtype=float)

	# pad_len = (WINDOW_LEN - len(datas) % WINDOW_LEN) % WINDOW_LEN
	# datas = np.pad(datas, (0, pad_len), mode="constant",
	# 			   constant_values=(np.nan,))
	# datas = datas.reshape((-1, WINDOW_LEN))
	#
	# y_vals = np.nanmean(datas, axis=1)
	# plt.plot(y_vals)

	print(len(datas))
	len_datas = len(datas)
	win = int(len_datas/WINDOW_LEN)
	print(win)
	win_datas = []

	for i in range(win):
		b, e = WINDOW_LEN*(i), WINDOW_LEN*(i+1)
		win_datas.append(np.mean(datas[b:e]))
	print(win_datas)
	data = np.asarray(win_datas, dtype=float)

	fig = plt.figure()
	# print(data)
	plt.plot(data)

	plt.title("Learning rate = {}".format(
				learning_rate))
	plt.xlabel("Window # ({} episodes each window)".format(WINDOW_LEN))
	plt.ylabel("Average Reward")

	plt.show()


def main(): 
	f = open(FILE_NAME, "r")
	data = parse_file(f)
	graph(data)


if __name__ == "__main__": 
	main()

