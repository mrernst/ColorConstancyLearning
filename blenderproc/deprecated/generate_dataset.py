import blenderproc as bproc
import numpy as np
import argparse
import os
import sys

file_path = os.path.abspath(os.path.dirname(__file__))
if file_path != sys.path[0]:
    sys.path.insert(0, file_path)
    os.chdir(file_path)
    del file_path

from utils import lights_position_rotation, set_lights


def main(args):
    
    light_powers = np.loadtxt('resources/light_powers.txt')
    light_colors = np.genfromtxt('resources/light_colors.txt',dtype='str')

    # position of the camera relative to the (0,0,0) point
    camera_location = [0, args.camera_radius, args.camera_height]
   
    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_obj(args.scene_dir + "/scene_cube_" + str(args.scene_id) + ".obj")

    # Set all entities from the scene as solid, passive objects
    for obj in objs:
        obj.enable_rigidbody(active=False)

    # Activate the physics simulation #why though? Shouldn't it be faster without?
    bproc.object.simulate_physics(
        min_simulation_time=4,
        max_simulation_time=20,
        check_object_interval=1
    )

    # Set how many frames your animation should contain
    bproc.utility.set_keyframe_render_interval(frame_end=args.n_frames)

    # add illuminators to the environment
    lights_list = []
    for _ in range(args.n_lights):
        light = bproc.types.Light()
        light.set_type("SPOT")
        lights_list.append(light)

    lights_position_rotation(lights_list, args=args)
    set_lights(lights_list, light_colors, light_powers, args=args)

    
    ## UNUSED - Find point of interest, all cam poses should look towards it
    #poi = bproc.object.compute_poi(objs)

    # define the camera resolution
    bproc.camera.set_resolution(args.camera_resolution, args.camera_resolution)

    # Set camera position and rotation
    cam_rotation_x = np.arctan2(camera_location[1], camera_location[2]) * -1
    cam_rotation_y = np.arctan2(camera_location[0], camera_location[2])
    cam_pose = bproc.math.build_transformation_mat(camera_location, [cam_rotation_x, cam_rotation_y, 0])
    bproc.camera.add_camera_pose(cam_pose)

    # render the whole pipeline
    data = bproc.renderer.render()

    # You could? additionally write the data to a .hdf5 container #this should be one container for all, if anything
    bproc.writer.write_hdf5(args.output_dir+"/images/"+str(args.scene_id)+"/", data)

    # write the animations into .gif files
    if args.gif_frame_duration > 0:
        bproc.writer.write_gif_animation(args.output_dir+"/gifs/"+str(args.scene_id)+"/", data, frame_duration_in_ms=args.gif_frame_duration,)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=int, default=0, help="ID number of cube in the scene")
    parser.add_argument('--scene_dir', type=str, default="resources", help="Path to the scene files (.obj)")
    parser.add_argument('--output_dir', type=str, default="dataset", help="Path to directory where final files will be saved")
    parser.add_argument('--n_frames', type=int, default="100", help="Set a global number of frames for your final animation")
    parser.add_argument('--n_lights', type=int, default="8", help="Number of illuminators to be added to the environment")
    parser.add_argument('--lights_radius', type=int, default="6", help="Radius of the circle around which the spotlights are placed")
    parser.add_argument('--lights_height', type=int, default="5", help="Height of spotlights from z=0 axis")
    parser.add_argument('--lights_min_power', type=int, default="300", help="Minimum power of the illuminators")
    parser.add_argument('--lights_max_power', type=int, default="1000", help="Maximum power of the illuminator")
    parser.add_argument('--camera_radius', type=int, default="-5", help="Radial distance of the camera")
    parser.add_argument('--camera_height', type=int, default="5", help="Height of the camera")
    parser.add_argument('--camera_resolution', type=int, default="64", help="Resolution of the camera")
    parser.add_argument('--gif_frame_duration', type=int, default="100", help="Duration in ms of each frame in gif")
    args = parser.parse_args()

    main(args)
