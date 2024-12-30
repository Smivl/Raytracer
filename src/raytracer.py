import numpy as np
import math
import config
from src.utils import *
from src.ImLite import *
from scipy.ndimage import gaussian_filter



class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end
class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0.,is_emissive = False, emissive_intensity = 0, emissive_color = None,k_a=None, k_translucency = 0, d_texture = None, refractive_index = 1):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        # Textures
        self.diffuse_texture : Image = d_texture
        
        # Diffuse, specular, mirror coefficients
        self.k_d = k_d if emissive_color is None else emissive_color
        self.k_s = k_s
        self.p = p
        self.k_m = 0 if is_emissive else k_m

        # Transparency/Refraction
        self.k_t = k_translucency
        self.refractive_index = refractive_index

        # Emissivity
        self.is_emissive = is_emissive
        self.emissive_intensity = emissive_intensity
        self.emissive_color = emissive_color

        # Ambient coeeff
        self.k_a = k_a if k_a is not None else k_d

        
    def get_diffuse(self, u, v):
        """Get diffuse coeff given (u,v) coords between 0,0 (top left) and 1,1 (bottom right)"""

        if (self.diffuse_texture == None): return self.k_d
        width, height = self.diffuse_texture.shape[:2]

        texX = int(u * (width - 1))
        texY = int(v * (height - 1))
        
        return self.diffuse_texture.pixels[texX, texY]
class Hit:

    def __init__(self, t,point=None, normal=None, material : Material=None, surface = None, t_exit = np.inf):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.surface = surface
        self.t_exit = t_exit

no_hit = Hit(np.inf)

# OBJECTS
class Sphere:

    def __init__(self, center, radius, material : Material, light_intensity = 0, u_texture_scale = 1, v_texture_scale = 1):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material
        self.light_intensity = light_intensity
        self.light_color = material.emissive_color
        self.u_texture_scale = 1/u_texture_scale
        self.v_texture_scale = 1/v_texture_scale

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """

        hitInfo = no_hit

        a = np.dot(ray.direction, ray.direction)
        oc = ray.origin - self.center
        b = 2*np.dot(oc, ray.direction)

        c = np.dot(oc, oc)-((self.radius)**2)

        discriminant = (b**2) - (4 * a * c)
        
        if (discriminant>0):
            t = (-b - np.sqrt(discriminant)) / (2*a)
    
            if(t>=0):
                intersect = ray.origin + ray.direction*t
                if(discriminant == 0):
                  hitInfo = Hit(t, intersect, normalize(intersect-self.center), self.material, self)
                else:
                  t2 = (-b + np.sqrt(discriminant)) / (2*a) # exiting time 
                  hitInfo = Hit(t, intersect, normalize(intersect-self.center), self.material, self, t2)

        return hitInfo
    
    def point_to_uv(self, point):
        # Calculate the direction from the center to the point
        direction = normalize(point - self.center)
        
        # Convert to spherical coordinates
        theta = np.arccos(direction[1]) # UP
        phi = np.arctan2(direction[0], direction[2])  # XZ plane
        
        #theta += np.pi/2 # Horizontal rotation
        #phi += np.pi/2  # Vertical rotation (optional)

        # Wrap angles to avoid overflow
        phi %= 2 * np.pi
        theta %= np.pi

        # Normalize spherical coordinates to UV space
        v = (theta / np.pi) * self.v_texture_scale
        u = (phi / (2*np.pi)) * self.u_texture_scale

        u += 0.45
        # Wrap UVs to stay within [0, 1]
        v %= 1.0
        u %= 1.0

        return (v, u)

    def get_emissive(self, point):
        if self.material.diffuse_texture != None:
          u,v = self.point_to_uv(point)
          col = self.material.get_diffuse(u, v)
        else:
          col = self.material.emissive_color
        return col
    
    def illuminate(self, ray : Ray, hit : Hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        point_on_sphere = self.center +(self.radius+1e-6)*normalize(self.center-hit.point)
 
        v = -normalize(ray.direction)
        l = normalize(point_on_sphere - hit.point)
        h = normalize(v+l)

        dist = np.dot(point_on_sphere - hit.point, point_on_sphere - hit.point)

        shadRay = Ray(hit.point + l * 1e-6, l)
        shad_Blocked = scene.intersect(shadRay, [self])

        if (shad_Blocked == no_hit and not hit.material.is_emissive):
            IntensityFalloff = ((max(0, np.dot(hit.normal, l)))/dist) * self.light_intensity  
            
            tex_u, tex_v = hit.surface.point_to_uv(hit.point)
            diffuse = hit.material.get_diffuse(tex_u, tex_v)

            DiffuseSpec = diffuse + (hit.material.k_s * ((np.dot(hit.normal,h))**hit.material.p))
            
            return DiffuseSpec * IntensityFalloff * self.light_color
          
        return 0
class Plane:
    def __init__(self, plane, material, texture_scale = 1.0):
        """Create a plane from given vertices

        Parameters:
          plane (4) -- an arry of 4 coefficients for the plane equation ax+by+cz+d = 0
          material : Material -- the material of the surface
        """
        self.plane = plane
        self.material = material
        self.texture_scale = texture_scale
        self.normal = vec(plane[:3])

    def point_to_uv(self, point):
        u = (point[0] / self.texture_scale) % 1
        v = (point[2] / self.texture_scale) % 1
        return (u, v)

    def intersect(self, ray : Ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        hitInfo = no_hit

        ray_origin_homogeneous = np.append(ray.origin, 1)  
        ray_direction_homogeneous = np.append(ray.direction, 0)  

        numerator = np.dot(self.plane.T, ray_origin_homogeneous)
        denominator = np.dot(self.plane.T, ray_direction_homogeneous)

        if denominator != 0:
          t = -(numerator/denominator)

          if t >= 0:
              intersect = ray.origin + ray.direction*t
              hitInfo = Hit(t, intersect, self.normal ,self.material, self)

        return hitInfo  
class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

    def point_to_uv(self, point):
        return (0,0)
    
    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        hitInfo = no_hit

        left = np.array([self.vs[0]-self.vs[1], self.vs[0]-self.vs[2], ray.direction]).T
        right = self.vs[0] - ray.origin

        try:
            beta, gamma, t = np.linalg.solve(left, right)
        except:
            return hitInfo

        normal = normalize(np.cross(self.vs[1] - self.vs[0], self.vs[2] - self.vs[0]))

        if beta >= 0 and gamma >= 0 and (beta + gamma) <= 1 and t > 0:
            intersect = ray.origin + ray.direction*t
            hitInfo = Hit(t, intersect, normal, self.material, self)

        return hitInfo

# CAMERA
class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        self.vfov = vfov
        self.target = target


        self.viewDirection = target - eye
        self.f = np.sqrt(np.dot(self.viewDirection, self.viewDirection)); # you should set this to the distance from your center of projection to the image plane
        self.M = np.eye(4);  # set this to the matrix that transforms your camera's coordinate system to world coordinates
        
        self.h = 2* np.tan(np.radians(self.vfov * 0.5)) * self.f
        self.w = self.h * self.aspect


        self.z = -normalize(self.viewDirection)
        self.x = normalize(np.cross(self.z, up))
        self.y = normalize(np.cross(self.z, self.x))


        self.M[:3, 0] = self.x
        self.M[:3, 1] = self.y
        self.M[:3, 2] = self.z
        self.M[:3, 3] = eye
        

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """

        x_trans = (img_point[0]- 0.5) * -self.w
        y_trans = (0.5-img_point[1]) * -self.h

        direction = self.M[:3, :3] @ np.array([x_trans, y_trans, -self.f]).T

        return Ray(self.eye, direction)
    
    def generate_sampling_ray(self, img_point):
        x_trans = (img_point[0]- 0.5) * -self.w
        y_trans = (0.5-img_point[1]) * -self.h


        direction = self.M[:3, :3] @ np.array([x_trans, y_trans, -self.f]).T

        return Ray(self.eye, direction)
    



# LIGHTS
class PointLight:

    def __init__(self, position, intensity, color = [1, 1, 1]):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity
        self.color = color

    def illuminate(self, ray : Ray, hit : Hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """

 
        v = -normalize(ray.direction)
        l = normalize(self.position - hit.point)
        h = normalize(v+l)

        dist = np.dot(self.position - hit.point, self.position - hit.point)

        shadRay = Ray(hit.point + l * 1e-6, l)
        shad_Blocked = scene.intersect(shadRay)
        

        if (shad_Blocked == no_hit and not hit.material.is_emissive):
            IntensityFalloff = ((max(0, np.dot(hit.normal, l)))/dist) * self.intensity  
            
            tex_u, tex_v = hit.surface.point_to_uv(hit.point)
            diffuse = hit.material.get_diffuse(tex_u, tex_v)

            DiffuseSpec = diffuse + (hit.material.k_s * ((np.dot(hit.normal,h))**hit.material.p))
          
            return DiffuseSpec * IntensityFalloff * self.color
            
        return 0
                

        # v = -normalize(ray.direction)
        # l = normalize(self.position - hit.point)
        # h = normalize(v + l)
        
        # diffuse = hit.material.k_d * max(0, np.dot(hit.normal, l))
        # specular = hit.material.k_s * (max(0, np.dot(hit.normal, h)) ** hit.material.p)
        # intensity_falloff = self.intensity / np.dot(self.position - hit.point, self.position - hit.point)
        # reflected_light = (diffuse + specular) * intensity_falloff
        # return reflected_light
class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        La = hit.material.k_a * self.intensity
        return La

# SCENE
class Scene:

    def __init__(self, surfs : list, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray, excluding = []):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        minHit = no_hit
        for surface in self.surfs:
            res = surface.intersect(ray)
            if(res != no_hit and res.t < minHit.t and surface not in excluding):
                minHit : Hit = res
        
        return minHit


# SKY BOX
textures = {
        'right': Image("resources/textures/right.png").GetFloatCopy(),
        'left': Image("resources/textures/left.png").GetFloatCopy(),
        'top': Image("resources/textures/top.png").GetFloatCopy(),
        'bottom': Image("resources/textures/bottom.png").GetFloatCopy(),
        'front': Image("resources/textures/front.png").GetFloatCopy(),
        'back': Image("resources/textures/back.png").GetFloatCopy()
}

def sample_skybox(direction):
    
    direction = direction / np.linalg.norm(direction)
    Dx, Dy, Dz = direction

    abs_dir = np.abs(direction)

    if abs_dir[0] > abs_dir[1] and abs_dir[0] > abs_dir[2]:  
        if Dx > 0:  # Right face
            v = 0.5 * (1 + Dz / Dx)
            u = 0.5 * (1 - Dy / Dx)
            texture = textures['left']
        else:  # Left face
            v = 0.5 * (1 + Dz / Dx)
            u = 0.5 * (1 + Dy / Dx)
            texture = textures['right']
    elif abs_dir[1] > abs_dir[0] and abs_dir[1] > abs_dir[2]: 
        if Dy > 0:  # Top face
            v = 0.5 * (1 + Dx / Dy)
            u = 0.5 * (1 - Dz / Dy)
            texture = textures['top']
        else:  # Bottom face 
            v = 0.5 * (1 - Dx / Dy)
            u = 0.5 * (1 - Dz / Dy)
            texture = textures['bottom']
    else: 
        if Dz > 0:  # Front face
            v = 0.5 * (1 - Dx / Dz)
            u = 0.5 * (1 - Dy / Dz)
            texture = textures['front']
        else:  # Back face
            v = 0.5 * (1 - Dx / Dz)
            u = 0.5 * (1 + Dy / Dz)
            texture = textures['back']

    height, width, _ = texture.shape
    x = min(int(u * width), width - 1)
    y = min(int(v * height), height - 1)

    return texture.pixels[x, y][:3]

# GAUSSIAN FILTER
"""
def gen_gaussian_filter(dimy , dimx , sigma):
    f = np.zeros([dimy, dimx])
    # A3TODO: Complete this function
    cx, cy = np.ceil(dimx//2), np.ceil(dimy//2)
    Cconstant = 1/(2*math.pi*(sigma**2))
    kconstant = -1/(2*(sigma**2))
    for y in range(dimy):
        for x in range(dimx):
            y_diff = y-cy
            x_diff = x-cx
            dist = x_diff**2 + y_diff**2
            value = Cconstant * math.exp(kconstant*dist)
            f[y,x]= value
    
    return f
"""

# REFLECT
def reflect(ray : Ray,hit : Hit):
    dir =  2 * np.dot(hit.normal, -ray.direction) * hit.normal + ray.direction
    return Ray(hit.point + dir * 1e-6, dir)

# REFRACTION
def refract(ray :Ray, hit: Hit, n_i, n_r):
    normal = hit.normal
    cos_theta1 = -np.dot(ray.direction, normal)

    if cos_theta1 < 0:
        # normal vector points towards ray
        normal = -normal
        cos_theta1 = -cos_theta1

    n = n_i / n_r
    sin_theta2_squared = n**2 * (1 - cos_theta1**2)

    # Total internal reflection
    if sin_theta2_squared > 1:
        return reflect(ray, hit)

    cos_theta2 = np.sqrt(1 - sin_theta2_squared)
    refracted_vector = normalize(n * ray.direction + (n * cos_theta1 - cos_theta2) * normal)

    return Ray(hit.point + refracted_vector* 1e-6, refracted_vector)

# SHADING
def shade(ray : Ray, hit : Hit, scene : Scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """

    if depth >= config.MAX_DEPTH:
      return 0
    else:

      
      contribution = 0

      # If material is emissive, the color of it should just be the raw emissive color times its intensity
      if hit.material.is_emissive:
          return hit.surface.get_emissive(hit.point) * hit.material.emissive_intensity

      # diffuse & specular & ambient for light sources
      for light in lights:
        contribution += light.illuminate(ray, hit, scene)    

      # if object is reflective
      if hit.material.k_m > 0:

        # generate new reflected ray
        r = reflect(ray, hit)
        newHit = scene.intersect(r)

        if newHit == no_hit:
          contribution += hit.material.k_m * sample_skybox(r.direction)
        else:
          Lr = shade(r, newHit, scene, lights, depth + 1)
          contribution += (hit.material.k_m * Lr)

      # if object is transparent
    
      if hit.material.k_t > 0:
        refracted_ray = refract(ray, hit, 1.0, hit.surface.material.refractive_index) if (hit.t_exit != np.inf) else refract(ray, hit, hit.surface.material.refractive_index,1.0)

     
        sec_hit : Hit = scene.intersect(refracted_ray)
        if sec_hit == no_hit:
            back_col = sample_skybox(refracted_ray.direction)
        else:
            back_col = shade(refracted_ray, sec_hit, scene, lights, 0)
          
        # amount of light that's not reflected at surface times the opacity (transparency).
        # AKA how much light is not absorbed from the light that is not reflected
        transmittance = hit.material.k_t * (1 - hit.material.k_m)
        # Final color is the contribution from the mirrored / absorbed times the amount of light not refracted
        # added to the amount of light refracted times the color returned by the refraction itself
        contribution = contribution * (1 - transmittance) + back_col * transmittance
      
      return contribution
    

# LUMINANCE FORMULA
def brightness(pixel):
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

# SUPERSAMPLING ROTATED GRID ALGORITHM
rot_angle = config.ROT_GRID_ANGLE
rot_matrix = np.array([[np.cos(rot_angle), np.sin(rot_angle)],[-np.sin(rot_angle), np.cos(rot_angle)]])
scale_factor = config.ROT_GRID_SCALE

def rotgrid(x, y):
  return np.dot(rot_matrix,np.array([x,y])) * scale_factor

# IMAGE RENDER
def render_image(camera : Camera, scene : Scene, lights : list, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    
    output_image = np.zeros((ny,nx,3), np.float32)
    
    emissivity_filter = np.zeros(output_image.shape, np.float32)
    post_process_bloom = np.zeros(output_image.shape, np.float32)

    total_pixels = ny*nx

    scaled_nx = nx
    scaled_ny = ny
    
    if config.AA_ALGORITHM != config.ANTI_ALIASING_ALGORITHM.NONE:
      scaled_nx = nx * config.SUPERSAMPLING_SCALE
      scaled_ny = ny * config.SUPERSAMPLING_SCALE

    for surf in scene.surfs:
        if surf.material.is_emissive and type(surf) != Triangle:
            lights.append(surf)

    for scaled_j in range(scaled_nx):
        for scaled_i in range(scaled_ny):

            j = scaled_j // config.SUPERSAMPLING_SCALE
            i = scaled_i // config.SUPERSAMPLING_SCALE
            

            if config.AA_ALGORITHM == config.ANTI_ALIASING_ALGORITHM.NONE:
              # NO SUPERSAMPLING
              x_scaled = (j+0.5)/nx
              y_scaled = (i+0.5)/ny

            elif config.AA_ALGORITHM == config.ANTI_ALIASING_ALGORITHM.GRID:
              # USE GRID SUPERSAMPLING
              x_scaled = (scaled_j + 0.5) / scaled_nx
              y_scaled = (scaled_i + 0.5) / scaled_ny

            elif config.AA_ALGORITHM == config.ANTI_ALIASING_ALGORITHM.ROTATED_GRID:
              # USE ROTATED GRID SUPERSAMPLING
              x_p = (((scaled_j + 0.5) - (j * config.SUPERSAMPLING_SCALE)) - 1)
              y_p = (((scaled_i + 0.5) - (i * config.SUPERSAMPLING_SCALE)) - 1)
              
              # Rotated and scaled sampling
              u, v = rotgrid(x_p, y_p)
              u /= config.SUPERSAMPLING_SCALE
              v /= config.SUPERSAMPLING_SCALE
              
              # Map to the high-res supersampled grid
              x_scaled = (u + (j * config.SUPERSAMPLING_SCALE)) / scaled_nx
              y_scaled = (v + (i * config.SUPERSAMPLING_SCALE)) / scaled_ny
              
            ray = camera.generate_sampling_ray([x_scaled, y_scaled])

            intersection = scene.intersect(ray)

            if(intersection == no_hit): 
              # skybox OR flat color
              raw_col = sample_skybox(ray.direction)
              #raw_col = scene.bg_color

            else:    
              #raw_col = 1
              #"""
              raw_col = shade(ray, intersection, scene, lights, 0)


              if(intersection.material.is_emissive):
                  emissivity_filter[i, j] += raw_col

                  if(brightness(normalize(raw_col)) > config.BRIGHTNESS_THRESHOLD):
                      post_process_bloom[i, j] += raw_col
            
  
              #"""
            output_image[i,j] += raw_col

        # Show progress
        print(f"Progress : {round((((j+1))/nx)*100,1)}% out of {total_pixels} pixels",end='\r')

    # Normalize
    output_image /= (config.SUPERSAMPLING_SCALE ** 2)
    emissivity_filter /= (config.SUPERSAMPLING_SCALE ** 2)
    post_process_bloom /= (config.SUPERSAMPLING_SCALE ** 2)

    # Apply Gaussian filter to each channel
    emissivity_mask = np.zeros_like(emissivity_filter)
    post_process_mask = np.zeros_like(post_process_bloom)

    sig_emissivity = min(11 * config.SUPERSAMPLING_SCALE, 11 * config.SUPERSAMPLING_SCALE)
    sig_post_process = min(35 * config.SUPERSAMPLING_SCALE, 35 * config.SUPERSAMPLING_SCALE)

    for i in range(3): 
        emissivity_mask[...,i] += gaussian_filter(emissivity_filter[...,i], sigma=sig_emissivity)
        post_process_mask[...,i] += gaussian_filter(post_process_bloom[...,i], sigma=sig_post_process)
    
    return output_image + emissivity_mask + post_process_mask
