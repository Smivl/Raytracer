from src.utils import *
from src.raytracer import *
from src.cli import render

def transform_obj(obj, scale = 1, roll = 0, yaw = 0, pitch = 0, translation = 0):
    res = obj * scale

    rotxy = np.array([[np.cos(roll), -np.sin(roll), 0],[np.sin(roll), np.cos(roll), 0],[0,0,1]])
    rotxz = np.array([[np.cos(yaw), 0, np.sin(yaw)],[0, 1, 0],[-np.sin(yaw),0,np.cos(yaw)]])
    rotyz = np.array([[1,0,0],[0,np.cos(pitch), -np.sin(pitch)],[0, np.sin(pitch), np.cos(pitch)]])

    res @= rotxy
    res @= rotxz
    res @= rotyz

    return res + translation


# MATERIALS

# SUN material
emissive_sun_col = Material(vec([0.0, 0.0, 1.0]),
               is_emissive = True,
               emissive_intensity = 5,
               emissive_color=vec([0.94,0.5,0.4]))
               #emissive_color=vec([0.4,1.0,0.4]))

# BLUE BALL mat
blue_ball = Material(vec([0.0, 0.0, 1.0]),
               is_emissive = True,
               emissive_intensity = 5,
               emissive_color=vec([0.0,0.0,1.0]))

# RED BALL mat
red_ball = Material(vec([0.0, 0.0, 1.0]),
               is_emissive = True,
               emissive_intensity = 5,
               emissive_color=vec([1.0,0.0,0.0]))

glass = Material(vec([1,1,0]), k_s=.4, p=90, k_m=0.1, k_translucency=0.95, refractive_index=1.25)
sand = Material(vec([0,0,0]), k_s=0.05, p=30, k_m=0.05, d_texture=Image(path="resources/textures/sand2.png").GetFloatCopy())
water = Material(vec([0,0,0]), k_s=0.02, p=30, k_m=0.05, d_texture=Image(path="resources/textures/clear-sea-water-2048x2048.jpg").GetFloatCopy())

bot_mat = Material(vec([0.6, 0.3, 0.1]), k_s=0.1, p=30, k_m = 0.05)
pole_mat = Material(vec([0.5, 0.3, 0.2]), k_s=0.1, p=30, k_m = 0.05)
sail_mat = Material(vec([0.9,0.9,0.8]), k_s=0.01, p=10, k_m = 0.0)

# UNUSED MATERIALS
blue = Material(vec([0.5, 0.5, 0.5]), k_s=0.5, p=90, k_m=0.2, k_translucency=1.0, refractive_index=1.125)
earth = Material(vec([0.5, 0.5, 0.5]), k_s=0.2, p=90, k_m=0.0, d_texture= Image(path="resources/textures/2k_earth_daymap.jpg").GetFloatCopy())

checker_board = Material(vec([0.0, 0.0, 0.0]), k_m=0.01, k_translucency= 0, d_texture= Image(path="resources/textures/Checkerboard-image-and-mesh.png").GetFloatCopy())

planet = Material(vec([1.0, 0.0, 0.0]),
                p=30, 
                k_m=0.00,
                is_emissive = True,
                emissive_intensity = 0.4,
                emissive_color=vec([0.31,0.407,0.525]),
                d_texture=Image(path="resources/textures/planet.jpg").GetFloatCopy())

boat_translation = [-4, 0.2, -20]

# Load and scale model
boat_bottom = transform_obj(read_obj_triangles(open("resources/models/boat_bottom.obj")), yaw = 50, translation=boat_translation)
boat_flagpole= transform_obj(read_obj_triangles(open("resources/models/boat_pole.obj")), yaw = 50, translation=boat_translation)
boat_sail = transform_obj(read_obj_triangles(open("resources/models/boat_sail.obj")), yaw = 50, translation=boat_translation)


scene = Scene([
    Sphere(vec([-5000,600,-10000]), 400, emissive_sun_col, 200000000),  #SUN!!
    
    #Sphere(vec([-9000,7800,-8000]), 2000, planet, 100, u_texture_scale= 0.5, v_texture_scale=1.0),  
    #Sphere(vec([0,0, -3]), 1, earth, u_texture_scale= 0.5, v_texture_scale=1.0), EARTH

    #Sphere(vec([0,-6.3,0]), 6.3, gray, u_texture_scale= 0.02, v_texture_scale=0.02),
    #Sphere(vec(boat_translation), 0.3, blue_ball, 4), # BOAT MARKER

    Sphere(vec([46,-498, -14]), 500, sand, u_texture_scale= 0.5, v_texture_scale=0.5),
    Sphere(vec([5,2.5,-10]), 0.2, blue_ball, 4),
    Sphere(vec([6.5,2.5,-8.5]), 0.2, red_ball, 4),
    Sphere(vec([8,2.5,-7]), 1, glass),
    Plane(vec([0,1,0,0]), water, texture_scale= 4)

] #+  [Triangle(tri, bot_mat) for tri in boat_bottom] + [Triangle(tri, pole_mat) for tri in boat_flagpole] + [Triangle(tri, sail_mat) for tri in boat_sail]
,vec([0,0,0]))

#vec([0.937254902, 0.321, 0.00]))
#vec([0.95, 0.3, 0.1])) sunset background color
cam_pos = vec([0,0.5,0])
target_pos = cam_pos + vec([0.0,0.0,-0.5])

lights = [
    #PointLight(cam_pos + vec([5,10,5]), vec([100,100,100]), vec([1,1,1])),
    #PointLight(vec([0,1,1]), vec([5,5,5]), vec([1,1,1])),
    AmbientLight(0.1),
]

#camera = Camera(vec([1,0.8,2]), target=vec([0,0.4,0]), vfov=70, aspect=16/9)
#camera = Camera(vec([3,1,5]), target=vec([0,0,0]), vfov=25, aspect=16/9)
camera = Camera(cam_pos, target=target_pos, vfov=90, aspect=16/9)


render(camera, scene, lights)