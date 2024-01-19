import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.style
import matplotlib as mpl

COLORSCHEME = "color"
PROJECTDIR = '/Users/markus/Desktop/'


from collections import namedtuple
from cycler import cycler

if COLORSCHEME == "monochrome":
	mpl.style.use('classic')
	#lines.lineStyles.keys()
	#markers.MarkerStyle.markers.keys()
	# Create cycler object. Use any styling from above you please
	monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '=.']) * cycler('marker', ['^',',', '.']))


	#bar_cycle = (cycler('hatch', ['///', '--', '...','\///', 'xxx', '\\\\']) * cycler('color', 'w')*cycler('zorder', [10]))
	bar_cycle = (cycler('hatch', ['/', '--', '...','\///', 'xx', '\\']) * cycler('color', 'w')*cycler('zorder', [10]))
	styles = bar_cycle()
else:
	mpl.style.use('default')
	#mpl.rcParams['errorbar.capsize'] = 3
	#mpl.rcParams['hatch.color'] = "white"
	#hatches = ["/////", '\\\\\\\\\\' , "/////"]
	#hatches = ["/////","/////","/////"]
	hatches = ["","",""]

	import seaborn as sns
	#current_palette = sns.cubehelix_palette(5, start=2.15, rot=-.005)
	#current_palette = sns.color_palette("hls", 5)
	#current_palette = sns.color_palette("Paired")
	#current_palette = sns.color_palette("colorblind")
	current_palette = sns.color_palette("Set1", 12)[3:5] + sns.color_palette("Set1", 12)[:3] + sns.color_palette("Set1", 12)[5:]
	#current_palette = sns.color_palette("Set1", 12)[3:]
	sns.set_palette(current_palette)
	sns.set_style("ticks")
	sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2})



INPUT = ["monocular"]
DBARRAY = ["obj.", "light."]
CONDARRAY = [r'${x}$ ', '$l_1$ ', '$l_2$ ', '$h$ ', '$z$ ']
CONDARRAY = [r'${x}$ ', '$l_1$ ', '$l_2$ ', '$l_3$ ', '$h$ ', '$z$ ']


ITERATIONS = 5


rects = []
n_groups = len(CONDARRAY)

data = np.zeros([ITERATIONS, len(DBARRAY), n_groups])
means = np.zeros([len(DBARRAY), n_groups])
stderror = np.zeros([len(DBARRAY), n_groups])


# get a list of all the relevant directories
temp_list = [
	'09-01-24_08:29_001b_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None',
	'09-01-24_09:27_001b_basicrun_2_seed_2_C3_aug_time_SimCLR_reg_None',
	'09-01-24_10:21_001b_basicrun_3_seed_3_C3_aug_time_SimCLR_reg_None',
	'09-01-24_11:15_001b_basicrun_4_seed_4_C3_aug_time_SimCLR_reg_None',
	'09-01-24_12:08_001b_basicrun_5_seed_5_C3_aug_time_SimCLR_reg_None'
]
rand_list = [
	'09-01-24_08:33_002b_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None',
	'09-01-24_09:31_002b_basicrun_2_seed_2_C3_aug_time_SimCLR_reg_None',
	'09-01-24_10:27_002b_basicrun_3_seed_3_C3_aug_time_SimCLR_reg_None',
	'09-01-24_11:22_002b_basicrun_4_seed_4_C3_aug_time_SimCLR_reg_None',
	'09-01-24_12:17_002b_basicrun_5_seed_5_C3_aug_time_SimCLR_reg_None'
]
jitter_list = [
	'09-01-24_15:00_010_colorjitter_1_seed_1_C3_aug_jitter_SimCLR_reg_None',
	'09-01-24_15:45_010_colorjitter_2_seed_2_C3_aug_jitter_SimCLR_reg_None',
	'09-01-24_16:32_010_colorjitter_3_seed_3_C3_aug_jitter_SimCLR_reg_None',
	'09-01-24_17:20_010_colorjitter_4_seed_4_C3_aug_jitter_SimCLR_reg_None',
	'09-01-24_18:08_010_colorjitter_5_seed_5_C3_aug_jitter_SimCLR_reg_None'
]

db_list = [rand_list, rand_list]
db_list2 = ['object_classifier', 'light_classifier']

layer_list = ['l0','l1','l2','l4','l5']
layer_list = ['l0','l1','l2','l3', 'l4','l5']

for s in range(ITERATIONS):
	for d in range(len(DBARRAY)):
		for l in range(len(CONDARRAY)):
			loaded = np.load(f'./save/{db_list[d][s]}/{db_list2[d]}/{layer_list[l]}_accloss_random.npy')
			print(loaded)
			data[s, d, l] = loaded[-1]
			
data[:,:,0] = None

# copy data from previous experiment
#data[:,0,4] = np.array([0.89130002, 0.85339999, 0.89569998, 0.88940001, 0.83609998])


means = np.mean(data, axis=0)
stderror = np.std(data, axis=0) #/ np.sqrt(ITERATIONS)
# fake std error
#stderror = np.array([[[.05, .03, .04, .06, .05]]])


fig, ax = plt.subplots(figsize=(4,3))


index = np.arange(n_groups)
bar_width = 0.3

opacity = 1
error_config = {'ecolor': '0.3'}

for k in range(len(DBARRAY)):
	print(k)
	if COLORSCHEME=="monochrome":
		rects.append(ax.bar(index + k*bar_width, means[k], bar_width,
		alpha=opacity, **next(styles),
		yerr=stderror[k], error_kw=error_config,
		label=DBARRAY[k], linewidth=0, edgecolor=current_palette[(k+1)]))
	else:
		rects.append(ax.bar(index + k*bar_width, means[k], bar_width,
		alpha=opacity,
		yerr=stderror[k], error_kw=error_config,
		label=DBARRAY[k], linewidth=0, edgecolor=current_palette[(k+1)], hatch=hatches[k]))




def lighten_color(color, amount=0.75):
	"""
	Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.
	
	Examples:
	>> lighten_color('g', 0.3)
	>> lighten_color('#F034A3', 0.6)
	>> lighten_color((.3,.55,.1), 0.5)
	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# plot the first bar separately
DBARRAY = ["temp.", "rand."]

data[:,:,0] = 1. - np.random.random([5,2])



# raw pixels in the rand dataset (obj)
data[0,0,0] = 0.4815
data[1,0,0] = 0.5115
data[2,0,0] = 0.4941
data[3,0,0] = 0.5068
data[4,0,0] = 0.4749

# raw pixels in the rand dataset (lighting)
data[0,1,0] = 0.9997
data[1,1,0] = 1.0000
data[2,1,0] = 0.9999
data[3,1,0] = 1.0000
data[4,1,0] = 1.0000

means = np.mean(data, axis=0)
stderror = np.std(data, axis=0) / np.sqrt(ITERATIONS)

for k in range(0,2):
	print(k)
	if COLORSCHEME=="monochrome":
		rects.append(ax.bar(0+k*bar_width, means[k,0], bar_width,
		alpha=opacity, **next(styles),
		yerr=stderror[k,0], error_kw=error_config, linewidth=0, edgecolor=lighten_color(current_palette[(k)])))
	else:
		rects.append(ax.bar(0+k*bar_width, means[k,0], bar_width,
		alpha=opacity,
		yerr=stderror[k,0], error_kw=error_config, linewidth=0, edgecolor=lighten_color(current_palette[(k)]), hatch="/////", color=current_palette[(k)]))

ax.set_xlabel('Network Layer')
ax.set_ylabel('Accuracy')
#ax.set_title('C3 (32x32), 100 epochs, 200 epochs fit', fontsize=10)






# ax.set_xticks(index + bar_width * 1)
#ax.set_ylim([0,1.2])
ax.set_xticks(index + bar_width * (len(DBARRAY)/2. - 0.5))
ax.set_xticklabels([cond[:-1] for cond in CONDARRAY])
ax.grid(axis='y', zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=False,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=True) # labels along the bottom edge are off
ax.spines['left'].set_visible(False)
#ax.axhline(y=linear_model_error, c='gray', linestyle='--')
#fig.tight_layout()
#plt.savefig('{}_{}_{}.jpg'.format(DBARRAY[i], CONDARRAY[j]))

ax.text(-0.18, 1.05, ['A','B','C','D','E','F'][1], transform=ax.transAxes, size=21, weight='bold')
fig.subplots_adjust(bottom=.18)
fig.subplots_adjust(left=.14)
ax.legend(frameon=False, facecolor='white', edgecolor='white', framealpha=1.0, fontsize=10, loc='upper right', bbox_to_anchor=(1, 1.19))
ax.set_ylim(0,1)
plt.show()



# second plot to see how you can add significance stars



groups=['Control','30min','24hour']
cell_lysate_avg=[11887.42595,   4862.429689,    3414.337554]
cell_lysate_sd=[1956.212855,    494.8437915,    525.8556207]
cell_lysate_avg=[i/1000 for i in cell_lysate_avg]
cell_lysate_sd=[i/1000 for i in cell_lysate_sd]


media_avg=[14763.71106,8597.475539,6374.732852]
media_sd=[240.8983759,  167.005365, 256.1374017]
media_avg=[i/1000 for i in media_avg] #to get ng/ml
media_sd=[i/1000 for i in media_sd]

fig, ax = plt.subplots()
index = numpy.arange(len(groups)) #where to put the bars
bar_width=0.45
opacity = 0.5
error_config = {'ecolor': '0.3'}

cell_lysate_plt=plt.bar(index,cell_lysate_avg,bar_width,alpha=opacity,color='black',yerr=cell_lysate_sd,error_kw=error_config,label='Cell Lysates')
media_plt=plt.bar(index+bar_width,media_avg,bar_width,alpha=opacity,color='green',yerr=media_sd,error_kw=error_config,label='Media')
plt.xlabel('Groups',fontsize=15)
plt.ylabel('ng/ml',fontsize=15)
plt.title('\n'.join(wrap('Average Over Biological Repeats for TIMP1 ELISA (n=3)',45)),fontsize=15)
plt.xticks(index + bar_width, groups)
plt.legend()
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)




# add significance stars

from matplotlib.markers import TICKDOWN

def significance_bar(start,end,height,displaystring,linewidth = 1.2,markersize = 8,boxpad  =0.3,fontsize = 15,color = 'k'):
	# draw a line with downticks at the ends
	plt.plot([start,end],[height]*2,'-',color = color,lw=linewidth,marker = TICKDOWN,markeredgewidth=linewidth,markersize = markersize)
	# draw the text with a bounding box covering up the line
	plt.text(0.5*(start+end),height,displaystring,ha = 'center',va='center',bbox=dict(facecolor='1.', edgecolor='none',boxstyle='Square,pad='+str(boxpad)),size = fontsize)

pvals = [0.001,0.1,0.00001]
offset  =1
for i,p in enumerate(pvals):
	if p>=0.05:
		displaystring = r'n.s.'
	elif p<0.0001:
		displaystring = r'***'
	elif p<0.001:
		displaystring = r'**'
	else:
		displaystring = r'*'

	height = offset +  max(cell_lysate_avg[i],media_avg[i])
	bar_centers = index[i] + numpy.array([0.5,1.5])*bar_width
	significance_bar(bar_centers[0],bar_centers[1],height,displaystring)
	

plt.savefig(PROJECTDIR + '/cocoa_barplot_{}.pdf'.format(COLORSCHEME))
plt.show()

plt.clf()
plt.close()
