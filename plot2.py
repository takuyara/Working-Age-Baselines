import matplotlib.pyplot as plt
values = [0.536980323, 0.532765447, 0.524837072, 0.529837072, 0.420866588, 0.526927981, 0.53709878, 0.524940968, 0.522733176, 0.538105077, 0.479934573, 0.535956316]
names = ["GraphAU(P)-SE", "GraphAU(F)-SE", "GraphAU(P)-LSTM", "GraphAU(F)-LSTM", "GraphAU(P)-MODE", "GraphAU(F)-AVERAGE", "ResNet(P)-SE", "ResNet(F)-SE", "ResNet(P)-LSTM", "ResNet(F)-LSTM", "ResNet(P)-MODE", "ResNet(F)-AVERAGE"]
colors = ["grey", "gold", "darkviolet", "turquoise", "r", "g", "b", "c", "m", "y", "k", "darkorange", "lightgreen", "plum", "tan", "k", "khaki", "pink", "skyblue", "lawngreen", "salmon"]
bot = 0.35
def draw_label(ax, bars, heights, bot):
	for b, h in zip(bars, heights):
		ax.annotate(f"{h:.4f}", xy = (b.get_x() + b.get_width() / 2, b.get_height() + bot), xytext = (0, 3), textcoords = "offset points", va = "bottom", ha = "center")
def pltit(datas, names):
	trans = lambda x, b : [t - b for t in x]
	fig, ax = plt.subplots(1, 1)
	bar = ax.bar(list(range(len(datas))), trans(datas, bot), tick_label = names, color = colors, bottom = bot)
	ax.set_xticklabels(names, rotation = 45, fontsize = 11)
	draw_label(ax, bar, datas, bot)
	# ax.get_yaxis().set_visible(False)
	# ax.set_title(f"The average valence and arousal classifcation performance of all baseline models")
	ax.set_ylabel("Accuracy", fontsize = 15)
	# ax.set_ylim(0.2, 0.8)
	plt.show()

pltit(values, names)
