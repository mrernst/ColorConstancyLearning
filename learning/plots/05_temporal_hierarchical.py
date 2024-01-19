import seaborn as sns
l_names = [r'${x}_t$', '$l_1$', '$l_2$', '$h$', '$z$']

#sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2})
markers = ["o", "v", "s", "D", "o", "v", "s", "D", "o", "v", "s", "D"]
#markers = ["o", "v", "^", "<", ">", "s", "D", "P", "X", "*"]
colors = sns.color_palette("colorblind", 12)
colors = sns.color_palette("tab10", 12)
colors = sns.color_palette()
colors = sns.color_palette("Set1", 12)


color_map_1 = sns.color_palette("Blues_r", 6)[1:]
#color_map_1 = sns.color_palette("ch:start=.2,rot=-.3")
#color_map_1 = sns.light_palette("steelblue")


color_map_2 = sns.color_palette("YlOrBr", 5)
color_map_2 = sns.color_palette("ch:s=-.2,r=.6")
color_map_2 = sns.light_palette("seagreen")
color_map_2 = sns.color_palette("Greens_r", 6)[1:]

# load the numpy file
object_accuracy_array_0 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/object_accuracy_200_0.npz.npy')
lighting_accuracy_array_0 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/lighting_accuracy_200_0.npz.npy')
object_accuracy_array_1 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/object_accuracy_200_1.npz.npy')
lighting_accuracy_array_1 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/lighting_accuracy_200_1.npz.npy')
object_accuracy_array_2 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/object_accuracy_200_2.npz.npy')
lighting_accuracy_array_2 = np.load('/Users/markus/Research/Code/ColorConstancyLearning/learning/main/lighting_accuracy_200_2.npz.npy')


object_accuracy_array = object_accuracy_array_0 + object_accuracy_array_1 + object_accuracy_array_2
lighting_accuracy_array = lighting_accuracy_array_0 + lighting_accuracy_array_1 + lighting_accuracy_array_2
print(object_accuracy_array)

object_accuracy_array_means = object_accuracy_array[:2].mean(0)
lighting_accuracy_array_means = lighting_accuracy_array[:2].mean(0)

object_accuracy_array_std = object_accuracy_array[:2].std(0)# / np.sqrt(2)
lighting_accuracy_array_std = lighting_accuracy_array[:2].std(0)# / np.sqrt(2)








fig, axes = plt.subplots(1,2,figsize=(4,3), sharex=True, sharey=True)



ax = axes[0]

ax.grid(axis='y', zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=True,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=True) # labels along the bottom edge are off
ax.spines['left'].set_visible(False)
ax.set_xlim(0,100)
ax.set_ylim(0,1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.text(-0.55, 1.05, ['A','B','C','D','E','F'][1], transform=ax.transAxes, size=21, weight='bold')	




ax.axhline(y=object_accuracy_array_means[:,0].mean(), xmin=0, xmax=100, color=sns.color_palette("Blues_r", 6)[0], label=r'${x}$ (obj.)', linestyle=':')
ax.fill_between([1,25,50,75,100], y1=object_accuracy_array_means[:,0]+object_accuracy_array_std[:,0], y2=object_accuracy_array_means[:,0]-object_accuracy_array_std[:,0],
	color=sns.color_palette("Blues_r", 6)[0], alpha=0.3)
for l in range(4):
	ax.plot([1,25,50,75,100], object_accuracy_array_means[:,l+1], label=f'{l_names[l+1]} (obj.)', color=color_map_1[l], marker=markers[l])
	ax.fill_between([1,25,50,75,100], y1=object_accuracy_array_means[:,l+1]+object_accuracy_array_std[:,l+1], y2=object_accuracy_array_means[:,l+1]-object_accuracy_array_std[:,l+1],
	color=color_map_1[l], alpha=0.3)

# Put a legend to the right of the current axis
#ax.legend(loc='center left', #bbox_to_anchor=(1, 0.5), 
#fontsize=10, frameon=False)




ax = axes[1]
	
ax.grid(axis='y', zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=True,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=True) # labels along the bottom edge are off
ax.spines['left'].set_visible(False)
ax.set_xlim(0,100)
ax.set_ylim(0,1)
ax.set_xlabel("Epoch")
#ax.set_ylabel("Accuracy")
#ax.text(-0.20, 1.05, ['A','B','C','D','E','F'][3], transform=ax.transAxes, size=21, weight='bold')	


ax.axhline(y=lighting_accuracy_array_means[:,0].mean(), xmin=0, xmax=100, color=sns.color_palette("Greens_r", 6)[0], label=r'${x}$ (light.)', linestyle=':')
ax.fill_between([1,25,50,75,100], y1=lighting_accuracy_array_means[:,0]+lighting_accuracy_array_std[:,0], y2=lighting_accuracy_array_means[:,0]-lighting_accuracy_array_std[:,0],
	color=sns.color_palette("Greens_r", 6)[0], alpha=0.3)
for l in range(4):
	ax.plot([1,25,50,75,100], lighting_accuracy_array_means[:,l+1], label=f'{l_names[l+1]}, (light.)', color=color_map_2[l],  marker=markers[l])
	ax.fill_between([1,25,50,75,100], y1=lighting_accuracy_array_means[:,l+1]+lighting_accuracy_array_std[:,l+1], y2=lighting_accuracy_array_means[:,l+1]-lighting_accuracy_array_std[:,l+1],
	color=color_map_2[l], alpha=0.3)

# # Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0+0.04, box.width * 0.8, box.height*0.96])

# Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
#fontsize=10, frameon=False)


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, bbox_to_anchor=(1, 0.95), 
fontsize=10, frameon=False)

fig.subplots_adjust(bottom=.18)
fig.subplots_adjust(left=.14)
fig.subplots_adjust(right=.70)

plt.savefig('plotb.pdf')
plt.show()