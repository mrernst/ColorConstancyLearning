import numpy as np
import time
import colorsys
from matplotlib import colors


COLOR_DICT = {
    'B': ([0.,0.,0.], 'black'),
    'W': ([1.,1.,1.], 'white'),
    'R': ([1.,0,0], 'red'),
    'B': ([0,1.,0], 'blue'),
    'G': ([0,0,1.], 'green'),
    'M': ([1.,0,1.], 'magenta'),
    'C': ([0,1.,1.], 'cyan'),
    'Y': ([1.,1.,0], 'yellow'),
}

def light_code_to_colorrgb(code):
    return COLOR_DICT[code][0]
    
def light_code_to_colorname(code):
    return COLOR_DICT[code][1]


def scale_lightness(colorname, scale_l, scale_s):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*colors.to_rgb(colorname))
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l),  min(1, s * scale_s))



   
# Function that calculates the rotation euler angles of each point with respect to the point (0,0,0)
def calculate_rotation_angles(x, y, z):
    angle_x_list = []
    angle_y_list = []

    for i in range(len(x)):
        angle_x = np.arctan2(y[i], z) * -1
        angle_x_list = np.append(angle_x_list, angle_x)
        angle_y = np.arctan2(x[i], z)
        angle_y_list = np.append(angle_y_list, angle_y)
    
    return angle_x_list, angle_y_list


# Function that calculates the position of each illuminator (0,0,0)
def lights_position_rotation(lights_list, args):
    theta = np.linspace(0, 2*np.pi, args.n_lights, endpoint=False)
    x = args.lights_radius * np.sin(theta)
    y = args.lights_radius * np.cos(theta)
   
    x_deg, y_deg = calculate_rotation_angles(x, y, z=args.lights_height)
    
    for i in range(args.n_lights):
        lights_list[i].set_location([x[i],y[i],args.lights_radius])
        lights_list[i].set_rotation_euler([x_deg[i],y_deg[i],0])


# Function that creates color combinations of spotlights in every frame
def set_lights(light_list, light_colors, light_powers, args):

    for frame in range(args.n_frames):
        for light_id, light in enumerate(light_list):
            light.set_energy(light_powers[frame, light_id], frame=frame)
            light.set_color(light_code_to_colorrgb(light_colors[frame][light_id]), frame=frame)
