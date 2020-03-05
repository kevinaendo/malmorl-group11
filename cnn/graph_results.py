import csv
import matplotlib.pyplot as plt 
import numpy as np

FILE_NAME = "results2.csv"
WINDOW_LEN = 50


def parse_file(f): 
	csv_reader = csv.reader(f, delimiter=",")

	learning_rate = -1
	gamma = -1
	rewards = []

	for index, value in enumerate(csv_reader): 
		# Get learning rate
		if index == 0: 
			learning_rate = value[0]

		# Get gamma 
		elif index == 1: 
			gamma = value[0]

		# Get rewards list
		else: 
			rewards.append(value[0])

	return (learning_rate, gamma, rewards)


def graph(data): 
	learning_rate = data[0]
	gamma = data[1] 

	fig = plt.figure()

	datas = np.asarray(data[2], dtype=float)

	pad_len = (WINDOW_LEN - len(datas) % WINDOW_LEN) % WINDOW_LEN
	datas = np.pad(datas, (0, pad_len), mode="constant", 
					constant_values=(np.nan,))
	datas = datas.reshape((-1, WINDOW_LEN))

	y_vals = np.nanmean(datas, axis=1)
	plt.plot(y_vals)

	plt.title("Learning rate = {} & Gamma = {}".format(
				learning_rate, gamma))
	plt.xlabel("Episode #")
	plt.ylabel("Reward")

	plt.show()


def main(): 
	f = open(FILE_NAME, "r")
	data = parse_file(f)
	graph(data)


if __name__ == "__main__": 
	main()

