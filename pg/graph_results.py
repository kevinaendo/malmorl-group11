import csv
import matplotlib.pyplot as plt 

FILE_NAME = "results.csv"


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
	plt.plot(data[2])

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

