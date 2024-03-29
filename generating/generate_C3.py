# import libraries
import numpy as np
import argparse
from matplotlib.colors import hsv_to_rgb
from utils import COLOR_DICT, light_code_to_colorname, scale_lightness
import matplotlib.pyplot as plt
import colorsys
import os
import shutil

def main(args):

    np.random.seed(12345)
    
    if not args.use_existing_resources:
        # hue circle
        if args.cube_colors == 'hue_circle':
            # 1. sample cube colors (this may happen only once, however it happens so fast that we can also repeat this)
            cube_rgb_colors = []
            #cube_hsv_colors = [(i * (360 / (args.n_cubes//2)), 100, 50) for i in range(args.n_cubes//2)] + \
            #	[(i * (360 / (args.n_cubes//2)), 50, 100) for i in range(args.n_cubes//2)]
            cube_hsv_colors = [(i * (360 / (args.n_cubes)), 50, 100) for i in range(args.n_cubes)]
            
            for color in cube_hsv_colors:
	            # print(color)
	            # print([color[0]/360, color[1]/100, color[2]/100])
	            cube_rgb_colors.append(hsv_to_rgb([color[0]/360, color[1]/100, color[2]/100]))
	            
            cube_rgb_colors = np.array(cube_rgb_colors)
            np.savetxt('resources/cubes.txt', cube_rgb_colors, fmt='%.3f')
        elif args.cube_colors == 'hue_sat_grid5':
            # hue sat variant 10*5
            cube_rgb_colors = []
            cube_hsv_colors = [(i * (360 / (args.n_cubes//5)), 90, 100) for i in range(args.n_cubes//5)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//5)), 70, 100) for i in range(args.n_cubes//5)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//5)), 50, 100) for i in range(args.n_cubes//5)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//5)), 30, 100) for i in range(args.n_cubes//5)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//5)), 10, 100) for i in range(args.n_cubes//5)]

            for color in cube_hsv_colors:
                # print(color)
                # print([color[0]/360, color[1]/100, color[2]/100])
                cube_rgb_colors.append(hsv_to_rgb([color[0]/360, color[1]/100, color[2]/100]))
                
            cube_rgb_colors = np.array(cube_rgb_colors)
            np.savetxt('resources/cubes.txt', cube_rgb_colors, fmt='%.3f')
        elif args.cube_colors == 'hue_sat_grid3':
            # hue sat variant 10*5
            cube_rgb_colors = []
            cube_hsv_colors = [(i * (360 / (args.n_cubes//3)), 50, 100) for i in range(args.n_cubes//3)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//3)), 60, 100) for i in range(args.n_cubes//3)]
            cube_hsv_colors += [(i * (360 / (args.n_cubes//3)), 40, 100) for i in range(args.n_cubes//3)]
            
            for color in cube_hsv_colors:
                # print(color)
                # print([color[0]/360, color[1]/100, color[2]/100])
                cube_rgb_colors.append(hsv_to_rgb([color[0]/360, color[1]/100, color[2]/100]))
                
            cube_rgb_colors = np.array(cube_rgb_colors)
            np.savetxt('resources/cubes.txt', cube_rgb_colors, fmt='%.3f')

        elif args.cube_colors == 'random':
            # random colors
            cube_rgb_colors = []
            cube_hsv_colors = [(int(np.random.random()*360), int(np.random.random()*100), int(np.random.random()*100)) for i in range(args.n_cubes)]
            for color in cube_hsv_colors:
                #print(color)
                # print([color[0]/360, color[1]/100, color[2]/100])
                cube_rgb_colors.append(hsv_to_rgb([color[0]/360, color[1]/100, color[2]/100]))
            cube_rgb_colors = np.array(cube_rgb_colors)
            np.savetxt('resources/cubes.txt', cube_rgb_colors, fmt='%.3f')
        else:
            sys.exit()

        # 2. setup lighting conditions (these should be different(?) for every object so save it --n_cubes times)
        
        
        # debug mode only renders 1 object
        args.n_cubes = 1 if args.debug else args.n_cubes
        
        # integrate the sinusoidal code you wrote for implementation
        possible_colors = np.array(['R', 'G', 'B', 'C', 'M', 'Y', 'W'])
        
        if args.neutral_lights:
            possible_colors =  np.array(['W'])
        
        if args.temporal:
            N_FRAMES = args.n_frames
            MAX_WATT = args.lights_max_power
            MIN_WATT = args.lights_min_power
            COLORS = possible_colors
            MAX_FRAMES = args.max_periodicity
            MIN_FRAMES = args.min_periodicity
            N_LIGHTS = args.n_lights
            OFFSET = 100
            if args.temporal_sampling:
                N_FRAMES = args.n_frames * 100
            
            for scene_id in range(args.n_cubes):
                dict_of_lights = {}
                if args.plot:
                    fig, ax = plt.subplots(N_LIGHTS,1, sharex=True, sharey=True, figsize=(10,4))
                
                for light in range(N_LIGHTS):
                    current_frame = 0
                    x_stacked = []
                    y_stacked = []
                    c_stacked = []
                    while current_frame <= N_FRAMES+OFFSET:
                        
                        color = np.random.choice(COLORS) if (np.random.random() > args.turn_on_probability) else 'O'
                        wattage = (np.random.random()*(1-MIN_WATT/MAX_WATT) + MIN_WATT/MAX_WATT) * MAX_WATT if color != 'O' else 0. # 300 -1000 Watt random luminosity
                        periodicity = int(np.random.random()*(MAX_FRAMES-MIN_FRAMES)+MIN_FRAMES) # randomly 3 to 50 frames
                        #x = np.arange(0, 2*np.pi, 0.1)
                        
                        x = np.arange(current_frame, current_frame+periodicity, 1)
                        y = np.sin((x-current_frame) * 2*np.pi/periodicity + 3*np.pi/2)*wattage/2 + wattage/2
                        
                        x_stacked.append(x)
                        y_stacked.append(y)
                        c_stacked.append([color]*x.shape[0])
                        
                        current_frame += periodicity
                        if args.plot:
                            ax[light].set_ylim([0-100,MAX_WATT + 100])
                            ax[light].set_xlim([OFFSET,N_FRAMES+OFFSET])
                            ax[light].spines['right'].set_visible(False)
                            ax[light].spines['top'].set_visible(False)
                            ax[light].spines['left'].set_visible(False)
                            #ax[light].spines['bottom'].set_visible(False)
                            ax[light].grid(axis='x', color='white', alpha=0.4)
                            ax[light].plot(x, y, color=scale_lightness(light_code_to_colorname(color), 0.75, 1.), alpha=1.0)
                            ax[light].set_xticks([])
                            ax[light].set_yticks([])
                    
                    x_stacked = np.concatenate(x_stacked)
                    y_stacked = np.concatenate(y_stacked)
                    c_stacked = np.concatenate(c_stacked)
                    
                    
                    
                    if args.temporal_sampling:
                        # sample n_frames random indices without putting back
                        random_indices = np.random.choice(len(x_stacked), args.n_frames, replace=False)
                        dict_of_lights[light] = [x_stacked[random_indices],y_stacked[random_indices],c_stacked[random_indices]]
                    else:
                        dict_of_lights[light] = [x_stacked[OFFSET:N_FRAMES+OFFSET],y_stacked[OFFSET:N_FRAMES+OFFSET],c_stacked[OFFSET:N_FRAMES+OFFSET]]

                
                if args.plot:
                    ax[0].set_ylim(0, args.lights_max_power)
                    #plt.tight_layout()
                    ax[0].set_title('Temporal Lighting Pattern')
                    ax[-1].set_xlabel('Time')
                    ax[-1].set_ylabel('Luminosity (W)')
                    ax[-1].yaxis.set_label_coords(-0.01,+4.5)
                    plt.show()
                else:
                    plt.clf()
                
                
                
                
                
                light_colors = np.array([dict_of_lights[c][-1] for c in range(args.n_lights)]).T
                light_powers = np.array([dict_of_lights[c][-2] for c in range(args.n_lights)]).T
                
                np.savetxt(f'resources/light_colors_{scene_id}.txt', light_colors, fmt='%s', delimiter='')
                np.savetxt(f'resources/light_powers_{scene_id}.txt', light_powers, fmt='%.3f')
			
        else:
            for scene_id in range(args.n_cubes):
                light_colors = np.random.choice(possible_colors, size=(args.n_frames, args.n_lights))
                light_powers = np.random.uniform(low=args.lights_min_power, high=args.lights_max_power, size=(args.n_frames, args.n_lights))
	                
                # Turn off individual lights with probability
                for row in range(args.n_frames):
                    for col in range(args.n_lights):
                        if np.random.random() > args.turn_on_probability:
                            light_colors[row, col] = 'O'
                            light_powers[row, col] = 0.0
                    if np.sum(light_powers[row,:]) == 0:
                        col = np.random.randint(low=0, high=args.n_lights)
                        light_colors[row, col] = np.random.choice(possible_colors)
                        light_powers[row, col] = np.random.uniform(low=args.lights_min_power, high=args.lights_max_power)
                
                np.savetxt(f'resources/light_colors_{scene_id}.txt', light_colors, fmt='%s', delimiter='')
                np.savetxt(f'resources/light_powers_{scene_id}.txt', light_powers, fmt='%.3f')
        
        
        if args.same_illumination:
            # if we want the same illumination for everything we overwrite every file once more with the latest
            # generated sequence
            for scene_id in range(args.n_cubes):
                np.savetxt(f'resources/light_colors_{scene_id}.txt', light_colors, fmt='%s', delimiter='')
                np.savetxt(f'resources/light_powers_{scene_id}.txt', light_powers, fmt='%.3f')
        
        # 3. build object files
        if args.ground_plane:
            gp_string = "--ground_plane"
        else:
            gp_string = "--no-ground_plane"
        
        bashCommand = f"blenderproc run export_objects.py {gp_string}" 
        os.system(bashCommand) #technically also only needs to be done once
        
    else:
        pass
    
    # 4. execute blenderproc script from this script
    #for scene_id in range(12,13,1): #range(args.n_cubes):
    for scene_id in range(args.n_cubes):
        bashCommand = f"blenderproc run bproc_generator.py --scene_id {scene_id} --n_frames {args.n_frames} --n_lights {args.n_lights} --camera_resolution {args.camera_resolution} --camera_radius {args.camera_radius} --camera_height {args.camera_height} --render_samples {args.render_samples} --gif_frame_duration {args.gif_frame_duration}"
        os.system(bashCommand)
    
    # 5. split the dataset into train, test and validation and copy them to folders accordingly
    
    # C3 - train | validate | test | label - 0 | 1 | ... | 50 - 0.hdf | ... | 5000.hdf
    # also copy the cubes and light.txt to the specific
    
    for mypath in ['./dataset/C3/train','./dataset/C3/test','./dataset/C3/val','./dataset/C3/labels',]:
        if not os.path.isdir(mypath):
            os.makedirs(mypath)
    
    whole_range = np.arange(0, args.n_frames, 1)
    train_range = whole_range[:int(args.n_frames*0.6)]
    val_range = whole_range[int(args.n_frames*0.6):int(args.n_frames*0.8)]
    test_range = whole_range[int(args.n_frames*0.8):]
    
    for scene_id in range(args.n_cubes):
        for image_number in train_range:
            destination_path = f'./dataset/C3/train/{scene_id:04}/'
            origin_path = f'./dataset/images/{scene_id:04}/{image_number:04}.png'
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(origin_path, destination_path)
        for image_number in val_range:
            destination_path = f'./dataset/C3/val/{scene_id:04}/'
            origin_path = f'./dataset/images/{scene_id:04}/{image_number:04}.png'
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(origin_path, destination_path)
        for image_number in test_range:
            destination_path = f'./dataset/C3/test/{scene_id:04}/'
            origin_path = f'./dataset/images/{scene_id:04}/{image_number:04}.png'
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(origin_path, destination_path)
        
        destination_path = f'./dataset/C3/labels/'
        origin_path = f'./resources/light_colors_{scene_id}.txt'
        shutil.copy2(origin_path, destination_path)
        
        destination_path = f'./dataset/C3/labels/'
        origin_path = f'./resources/light_powers_{scene_id}.txt'
        shutil.copy2(origin_path, destination_path)
    destination_path = f'./dataset/C3/labels/'
    origin_path = f'./resources/cubes.txt'
    shutil.copy2(origin_path, destination_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cubes', type=int, default="50", help="Number of different cubes (needs to be even)")
    parser.add_argument('--n_lights', type=int, default="8", help="Number of illuminators to be added to the environment")
    parser.add_argument('--n_frames', type=int, default="1000", help="Number of different illumination combinations")
    parser.add_argument('--lights_min_power', type=int, default="300", help="Minimum power of the illuminators")
    parser.add_argument('--lights_max_power', type=int, default="1000", help="Maximum power of the illuminator")
    parser.add_argument('--max_periodicity', type=int, default="9", help="maximum frames for lights cycle")
    parser.add_argument('--min_periodicity', type=int, default="3", help="minimum frames for lights cycle")
    parser.add_argument('--camera_resolution', type=int, default="32", help="Resolution of the camera")
    parser.add_argument('--camera_radius', type=float, default="-4", help="Radial distance of the camera")
    parser.add_argument('--camera_height', type=float, default="4", help="Height of the camera")
    parser.add_argument('--render_samples', type=int, default="64", help="number of samples used for rendering the light passes")
    parser.add_argument('--gif_frame_duration', type=int, default="0", help="Duration in ms of each frame in gif")
    parser.add_argument('--illumination_id', type=int, default=-1, help="Illumination id so that each object gets illuminated the same way")
    parser.add_argument('--turn_on_probability', type=float, default="0.5", help="probability with with the lights turn on")
    parser.add_argument('--cube_colors', type=str, default='hue_circle', choices=["hue_circle", "random", "hue_sat_grid5", "hue_sat_grid3"], help="Different ways to sample the cube colors")

    
    # boolean arguments
    
    parser.add_argument('--use_existing_resources', dest='use_existing_resources', action='store_true', help="Use existing resource files, i.e. don't generate them")
    parser.add_argument('--no-use_existing_resources', dest='use_existing_resources', action='store_false')
    parser.set_defaults(use_existing_resources=False) # should be default true on release I think
    
    
    parser.add_argument('--plot', dest='plot', action='store_true', help="Plot an overview over the scene")
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=False)
    
    parser.add_argument('--temporal', dest='temporal', action='store_true', help="Make a temporally consistent dataset")
    parser.add_argument('--no-temporal', dest='temporal', action='store_false')
    parser.set_defaults(temporal=True)
    
    parser.add_argument('--temporal_sampling', dest='temporal_sampling', action='store_true', help="Make a temporally consistent dataset")
    parser.add_argument('--no-temporal_sampling', dest='temporal_sampling', action='store_false')
    parser.set_defaults(temporal_sampling=False)
    
    parser.add_argument('--neutral_lights', dest='neutral_lights', action='store_true', help="Force all lights in the scene to be white")
    parser.add_argument('--no-neutral_lights', dest='neutral_lights', action='store_false')
    parser.set_defaults(neutral_lights=False)
    
    parser.add_argument('--debug', dest='debug', action='store_true', help="debug mode only renders one object")
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)
    
    parser.add_argument('--same_illumination', dest='same_illumination', action='store_true', help="use the same illumination for every object")
    parser.add_argument('--no-same_illumination', dest='same_illumination', action='store_false')
    parser.set_defaults(same_illumination=False)
    
    parser.add_argument('--ground_plane', dest='ground_plane', action='store_true', help="Have a ground plane on the images")
    parser.add_argument('--no-ground_plane', dest='ground_plane', action='store_false')
    parser.set_defaults(ground_plane=True)
    
    
    
    
    args = parser.parse_args()
    
    main(args)