import blenderproc as bproc
from math import sqrt
import numpy as np
import os
import argparse
import bpy


def main(args):
	bpy.context.scene.render.engine = 'CYCLES'
	
	file_path = bpy.data.filepath
	directory =  os.path.dirname(os.path.realpath(__file__)) # os.path.dirname(file_path)
	
	print(file_path)
	print(directory)
	
	# remove the default cube
	try:
		cube = bpy.data.objects['Cube']
		bpy.data.objects.remove(cube, do_unlink=True)
	except:
		print("Object bpy.data.objects['Cube'] not found")
	
	
	
	bpy.ops.outliner.orphans_purge()
	
	### FLOOR
	# try to remove the ground plane (dependent on a cmd-line argument)
	if args.ground_plane:
		floor_verts = [
			(-10.0, -10.0, -0.001),
			(-10.0, 10.0, -0.001),
			(10.0, 10.0, -0.001),
			(10.0, -10.0, -0.001),
		]
		
		floor_faces = [
			(0, 1, 2, 3),
		]
		
		floor_edges = []
		
		floor_mesh = bpy.data.meshes.new("floor_data")
		floor_mesh.from_pydata(floor_verts, floor_edges, floor_faces)
		
		floor_obj = bpy.data.objects.new("floor_object", floor_mesh)
		
		floor_mat = bpy.data.materials.new("floor_material")
		floor_mat.use_nodes = True
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Sheen Tint'].default_value = 0.5
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Sheen'].default_value = 0.0
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Clearcoat'].default_value = 0.0
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Clearcoat Roughness'].default_value = 0.03
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.0
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Transmission'].default_value = 0.0
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['Transmission Roughness'].default_value = 0.0
		floor_mat.node_tree.nodes['Principled BSDF'].inputs['IOR'].default_value = 1.45
		
		floor_obj.active_material = floor_mat
		bpy.context.collection.objects.link(floor_obj)
	else:
		pass
	
	### CUBE
	
	cube_verts = [
		(0, -sqrt(2), 0.0),
		(-sqrt(2), 0, 0.0),
		(0, sqrt(2), 0.0),
		(sqrt(2), 0, 0.0),
		(0, -sqrt(2), 2.0),
		(-sqrt(2), 0, 2.0),
		(0, sqrt(2), 2.0),
		(sqrt(2), 0, 2.0),
	]
	
	cube_faces = [
		(0, 1, 2, 3),
		(7, 6, 5, 4),
		(4, 5, 1, 0),
		(7, 4, 0, 3),
		(6, 7, 3, 2),
		(5, 6, 2, 1),
	]
	
	cube_edges = []
	
	cube_mesh = bpy.data.meshes.new("cube_data")
	cube_mesh.from_pydata(cube_verts, cube_edges, cube_faces)
	
	cube_obj = bpy.data.objects.new("cube_object", cube_mesh)
	
	cube_mat = bpy.data.materials.new("cube_material")
	cube_mat.use_nodes = True
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (1, 1, 1, 1)
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Specular'].default_value = 0.5
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.5
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Sheen Tint'].default_value = 0.5
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Sheen'].default_value = 0.0
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Clearcoat'].default_value = 0.0
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Clearcoat Roughness'].default_value = 0.03
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.0
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Transmission'].default_value = 0.0
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['Transmission Roughness'].default_value = 0.0
	cube_mat.node_tree.nodes['Principled BSDF'].inputs['IOR'].default_value = 1.45
	
	cube_obj.active_material = cube_mat
	bpy.context.collection.objects.link(cube_obj)
	
	
	### LOOP THROUGH CUBE COLORS AND EXPORT OBJ FILES
	cube_colors = np.loadtxt(directory+'/resources/cubes.txt')
	
	for cube_id in range(cube_colors.shape[0]):
		
		cube_mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (
			cube_colors[cube_id,0],
			cube_colors[cube_id,1],
			cube_colors[cube_id,2],
			1)
		cube_obj.active_material = cube_mat
	
	
		### EXPORT OBJ FILE
		
		bpy.ops.export_scene.obj(filepath=directory+'/resources/scene_cube_'+str(cube_id)+'.obj')


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--ground_plane', dest='ground_plane', action='store_true', help="Have a ground plane on the images")
	parser.add_argument('--no-ground_plane', dest='ground_plane', action='store_false')
	parser.set_defaults(ground_plane=True)
	
	args = parser.parse_args()
	
	main(args)
