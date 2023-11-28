import numpy as np
import time

# Converts light O,W,A-I codes into RGB colors
# using the base colors of the illuminators
def light_code_to_color(code):
    if code == 'O':
        return [0, 0, 0]
    if code == 'W':
        return [1, 1, 1]
    if code == 'A':
         return [1.0, 1.0, 0.6]
    if code == 'B':
         return [0.678, 0.847, 0.902]
    if code == 'C':
         return [1, 0, 0]
    if code == 'D':
         return [0, 1, 0]
    if code == 'E':
         return [0, 0, 1]
    if code == 'F':
         return [1, 1, 0]
    if code == 'G':
         return [1, 0, 1]
    if code == 'H':
         return [0, 1, 1]

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
            light.set_color(light_code_to_color(light_colors[frame][light_id]), frame=frame)

    

    '''
    seed_value = int(time.time()) + frame
    np.random.seed(seed_value)

    rand_int = np.random.randint(0, 7)

    arr = np.arange(1, 9)
    np.random.shuffle(arr)

    selected_numbers = arr[:2]
    arr = np.delete(arr, [0, 1])

    rand_arr = arr[:rand_int]
    np.random.shuffle(rand_arr)

    arr_energy = np.random.uniform(args.lights_min_power, args.lights_max_power, rand_int + 2)

    for frame in range(args.n_frames):
        for i, light in enumerate(light_list):
            if light_index in rand_arr or light_index in selected_numbers:
                light.set_energy(arr_energy[0], frame=frame)
                seed_value = int(time.time()) + frame * (i + 2)
                if light_index in selected_numbers:
                    light.set_color(color_white)
                else:
                    np.random.seed(seed_value)
                    np.random.shuffle(color_list)

                    color_s = np.random.randint(0, len(color_list))
                    light.set_color(color_list[color_s])
                arr_energy = np.delete(arr_energy, 0)
            else:
                light.set_energy(0, frame=frame)
    '''