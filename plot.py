import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os.path



def main():
	with open("data_to_plot.dat","r") as file_in:
		all_data = file_in.readlines()

	all_data_nums = []
	for i in range(0, len(all_data)):
		tmp = all_data[i].strip().split(" ")
		tmp[0] = int(tmp[0])	# epoch
		tmp[1] = int(tmp[1])	# label
		tmp[2] = int(tmp[2])	# sample

		tmp[3] = float(tmp[3]) 	# x
		tmp[4] = float(tmp[4])	# y

		all_data_nums.append(tmp)

	num_epoch = all_data_nums[-1][0] + 1

	class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
	cmap 		 = ["red", "green", "blue", "yellow", "aqua", "navy", "maroon", "magenta", "orange", "crimson"]
	for i in range(0, num_epoch):
		if os.path.isfile(f"./pictures/plot_epochs/epoch_{i}.png") == True:
			continue

		this_epoch_list = [all_data_nums[j] for j in range(len(all_data_nums)) if all_data_nums[j][0] == i]

		figure(figsize=(12, 10), dpi=80)
		plt.grid()

		plt.xlim(-60, 60)
		plt.ylim(-130, 90)

		# Yes, it's 3-nested loop inside outer loop but it is fast enough, only 10 items each and most of them make 'continue'
		for label in range(0, 10):		
			x = []
			y = []

			for entry in this_epoch_list:
				if entry[1] != label:
					continue
		
				for sample in range(0, 10):
					if entry[2] != sample:
						continue

					x.append(entry[3])
					y.append(entry[4])

			plt.scatter(x, y, c=cmap[label], label=class_labels[label])

		plt.title(f"Epoch {i}", fontsize=24)
		plt.legend(loc ="upper left")
		plt.savefig(f"./pictures/plot_epochs/epoch_{i}.png")
		plt.close()




if __name__ == '__main__':
	main()
