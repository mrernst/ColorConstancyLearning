import numpy as np
import argparse

def main(args):
    
    # CUBES - Saves n_cubes rgb combinations 

    cubes = np.random.uniform(low=0, high=1, size=(args.n_cubes, 3))
    np.savetxt('resources/cubes.txt', cubes, fmt='%.3f')


    # LIGHTS - Saves n_frames combinations of n_lights values,
    # with O meaning the light is off, W meaning the light is white,
    # and A-I meaning the light is one of the base colors.
    # The probability of each light being on is 0.5, and if it's on
    # the color and intensity are sampled randomly

    possible_colors = np.array(['W','A','B','C','D','E','F','G','H'])

    light_colors = np.random.choice(possible_colors, size=(args.n_frames, args.n_lights))
    light_powers = np.random.uniform(low=args.lights_min_power, high=args.lights_max_power, size=(args.n_frames, args.n_lights))

    # Turn off individual lights with probability 0.5
    for row in range(args.n_frames):
        for col in range(args.n_lights):
            if np.random.random() < 0.5:
                light_colors[row, col] = 'O'
                light_powers[row, col] = 0.0
        if np.sum(light_powers[row,:]) == 0:
            col = np.random.randint(low=0, high=args.n_lights)
            light_colors[row, col] = np.random.choice(possible_colors)
            light_powers[row, col] = np.random.uniform(low=args.lights_min_power, high=args.lights_max_power)
            
    np.savetxt('resources/light_colors.txt', light_colors, fmt='%s', delimiter='')
    np.savetxt('resources/light_powers.txt', light_powers, fmt='%.3f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cubes', type=int, default="50", help="Number of different cubes")
    parser.add_argument('--n_lights', type=int, default="8", help="Number of illuminators to be added to the environment")
    parser.add_argument('--n_frames', type=int, default="1000", help="Number of different illumination combinations")
    parser.add_argument('--lights_min_power', type=int, default="300", help="Minimum power of the illuminators")
    parser.add_argument('--lights_max_power', type=int, default="1000", help="Maximum power of the illuminator")
    args = parser.parse_args()

    main(args)