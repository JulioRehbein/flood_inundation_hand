import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import deque

def conv(mat,kernel):
    #assuming a square odd dimmesion kernel
    mat = mat.astype(np.float64)
    mat = np.pad(mat,((1,1),(1,1)),mode = "edge")
    
    result = np.zeros_like(mat)
    mrow,mcol = mat.shape
    ksize= kernel.shape[0]
    koff = (ksize - 1) // 2
    for mi in range(koff,mrow-koff):
        for mj in range(koff,mcol-koff):
            accumulated = 0
            for ki in range(ksize):
                for kj in range(ksize):
                    row = mi - koff + ki
                    col = mj - koff + kj
                    if (0 <= row < mrow) and (0 <= col < mcol):
                        accumulated += mat[row,col] * kernel[ki,kj]
            result[mi,mj] = accumulated
    result = result[1:result.shape[0]-1,1:result.shape[1]-1]
    return result

def pyramid(nlayers):
    maxval = nlayers
    ndim = 2*nlayers -1
    minval = maxval - nlayers + 1
    result = np.empty((ndim,ndim))
    for i in range(ndim//2 + 1):
        for j in range(ndim//2 +1):
            if (j < i + 1):
                val = minval + j
            else:
                val = minval + i
                
            result[i,j] = val
            result[i,ndim-j-1] = val
            result[ndim-i-1,j] = val
            result[ndim-i-1,ndim-j-1] = val
    return result


def hemisphere(radius):
    ndim = 2*radius - 1
    result = np.zeros((ndim,ndim))
    for z in range(radius):
        newradius = int(np.sqrt(radius**2 - z**2))
        for i in range(ndim//2 + 1):
            for j in range(ndim//2 +1):
                dx = np.abs(radius - i -1)
                dz = np.abs(radius - j -1)
                rad = np.sqrt(dx**2 + dz**2)
                if rad <= newradius:
                    result[i,j] = z
                    result[i,ndim-j-1] = z
                    result[ndim-i-1,j] = z
                    result[ndim-i-1,ndim-j-1] = z
    return result


def paraboloid(radius,height):
    ndim = 2*radius - 1
    result = np.zeros((ndim,ndim))
    #latus_rectum = radius**2/height
    for h in range(height):
        hnew = height - h
        newradius = radius * np.sqrt(hnew/height)
        for i in range(ndim//2 + 1):
            for j in range(ndim//2 +1):
                dx = np.abs(radius - i -1)
                dz = np.abs(radius - j -1)
                rad = np.sqrt(dx**2 + dz**2)
                if rad <= newradius:
                    result[i,j] = h
                    result[i,ndim-j-1] = h
                    result[ndim-i-1,j] = h
                    result[ndim-i-1,ndim-j-1] = h
    return result

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def voxel_plot(xs,ys,zs,ax):
    row,col = zs.shape
    dx = np.abs(xs[0,1] - xs[0,0]) * np.ones(xs.size)
    dy = np.abs(ys[1,0] - ys[0,0]) * np.ones(ys.size)
    xs = xs.ravel()
    ys = ys.ravel()
    maxdiffs = []
    for i in range(row):
        for j in range(col):
            value = zs[i,j]
            try:
                top_value = zs[i-1,j]
            except:
                top_value = value
            try:
                bottom_value = zs[i+1,j]
            except:
                bottom_value = value
            try:
                left_value = zs[i,j-1]
            except:
                left_value = value
            try:
                right_value = zs[i,j+1]
            except:
                right_value = value

            diffs = [value - val for val in [top_value,bottom_value,left_value,right_value]]
            maxdiffs.append(max(diffs))
    maxdiffs = np.array(maxdiffs)
    offsets = zs.ravel() - maxdiffs
    
    normalizer = Normalize(vmin = zs.min(),vmax = zs.max())
    color_values = cm.jet(normalizer(zs.ravel()))
    ax.bar3d(xs, ys, offsets, dx, dy, maxdiffs, color = color_values)


def fill_sinks(dem, threshold, max_iterations=100):
    filled_dem = dem.copy()
    rows, cols = dem.shape
    
    for i in range(max_iterations):
        updated_cells = 0
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                neighbors = [
                    filled_dem[r-1, c],
                    filled_dem[r+1, c],
                    filled_dem[r, c-1],
                    filled_dem[r, c+1],
                    
                    filled_dem[r-1,c-1],
                    filled_dem[r+1,c-1],
                    filled_dem[r+1,c-1],
                    filled_dem[r+1,c+1]
                ]
                
                min_neighbor = min(neighbors)
                
                if filled_dem[r, c] > min_neighbor + threshold:
                    filled_dem[r, c] = min_neighbor + threshold
                    updated_cells += 1
        
        if updated_cells == 0:
            break
        print(f"Filling sinks: {i+1} of {max_iterations}.")
    
    return filled_dem



def normalize(x,newmin,newmax,oldmin,oldmax):
    prop = (x - oldmin)/(oldmax - oldmin)
    return (newmax-newmin)*prop + newmin



def get_neighbors(theta,method ="dinf"):
    #this was reversed -- use when appropriate
    DIRECTIONS = {0:(0,-1),45:(1,-1),90:(1,0),135:(1,1),
                          180:(0,1),225:(-1,1),270:(-1,0),315:(-1,-1),
                          360:(0,-1)}
    
    DIRECTIONS = {0:(0,1),45:(1,1),90:(1,0),135:(1,-1),
                  180:(0,-1),225:(-1,-1),270:(-1,0),315:(-1,1),360:(0,-1)}
    
    
    #flat cells containing no aspect or edge cells not included by the kernel
    if np.isnan(theta) or method == "mfd":
        #return all directions (to be filtered later: reject higher elevations)
        angles,vecs = [],[]
        for key,val in DIRECTIONS.items():
            if key != 360:
                angles.append(key)
                vecs.append(val)
        return (tuple(angles),tuple(vecs))
    #wrap back
    if theta > 360:
        theta = theta % 360
    
    #single cell direction
    if theta % 45 == 0:
        return (theta,),(DIRECTIONS[int(theta)],)
    else:
        #two cell DIRECTIONS
        theta1 = int(45*np.floor(theta/45)) # closest going ccw
        theta2 = int(45*np.ceil(theta/45)) #closest going cw    
        return (theta1,theta2),(DIRECTIONS[theta1],DIRECTIONS[theta2])

class Cell():
    def __init__(self,row,col,aspect,elevation):
        self.row = row
        self.col = col
        self.aspect = aspect
        self.accum = 1
        self.weight = 1
        self.sources = 1
        self.is_sink = False
        self.elevation = elevation
        self.is_flat = True if np.isnan(aspect) else False
        self.neighbors = []
        self.neighbor_angles = []
        self.neighbor_proportions = []
        self.is_explored = False
        self.path_id = None
        self.parents = []
        self.strahler = 1
     
    def update_strahler(self,parent_strahler):
        if parent_strahler > self.strahler:
            self.strahler = parent_strahler
        elif parent_strahler == self.strahler:
            self.strahler += 1
        else:
            #do nothing
            pass
    
    def propagate_to(self,neighbor):
        neighbor.update_strahler(self.strahler)
        
    
    def _add_source(self,parent):
        self.sources += parent.sources
    
    def _flowin(self,amount,parent):
        if not np.isnan(amount):
            self.accum += amount
            self.parents.append(parent)
        
    def flowout(self,neighbor,proportion):
        amount = proportion * self.accum * self.weight
        neighbor._flowin(amount,self)
        neighbor._add_source(self)
        
    def flowall(self):
        if len(self.neighbors) == 0:
            raise Exception("Set cell neighbor first.")
            return
        for neighbor,proportion in zip(self.neighbors,self.neighbor_proportions):
            self.flowout(neighbor,proportion)

    def explore(self,path_id):
        self.is_explored = True
        self.path_id = path_id
    
    def __repr__(self):
        return f"""row: {self.row}\ncol: {self.col}\naccum: {self.accum}\nweight: {self.weight}\naspect: {self.aspect}\nelevation: {self.elevation}"""        

def get_flow_proportion(cell,method = "dinf",p = 1.1,dem_resolution = 30):
    # p and dem_resolution only gets used for mfd
    
    if method == "dinf":
        if not cell.is_flat:
            #if there is only a single neighbor, it can only flow to that
            if len(cell.neighbor_angles) == 1:
                if cell.aspect == cell.neighbor_angles[0]:
                    return [1]
            
            diffs = []
            for theta in cell.neighbor_angles:
                diffs.append(abs(cell.aspect-theta))
            proportions = [(sum(diffs) - diff)/sum(diffs) for diff in diffs]
        else:
            if not cell.is_sink:
                proportions = [1/len(cell.neighbors)] * len(cell.neighbors)
            else:
                return []
        return proportions
    
    elif method == "mfd":
        if not cell.is_flat and not cell.is_sink:
            
            #if there is only a single neighbor, it can only flow to that
            if len(cell.neighbor_angles) == 1:
                #aspect is not considered anymore unlike dinf
                return [1]
            
            
            # keep in mind that for mfd, the neighbors will have to be the
            # all cells that are lower than the current. This unlike the dinf
            # method where only two cells are considered as neighbors at a time
            
            #Note: angle is the azimuthal angle of the neighbor relative to the cell
            slopes = []
            diagonals = (45,135,225,315)
            for neighbor,angle in zip(cell.neighbors,cell.neighbor_angles):
                if angle in diagonals:
                    multiplier = 1/(dem_resolution* np.sqrt(2))
                else:
                    multiplier = 1/dem_resolution
                slope = cell.elevation - neighbor.elevation
                slope *= multiplier
                slopes.append(slope**p)
            if sum(slopes) > 0:
                proportions = [slope/sum(slopes) for slope in slopes]
            else:
                # this happens when all neighbors are of the same elevation as
                # the current cell. Thus sum(slopes) = 0.
                # We divide the flow equally among neighbors as an alternative
                proportions = [1/len(cell.neighbors)] * len(cell.neighbors)
                
            return proportions
        else:
            #flat not sink
            if not cell.is_sink:
                proportions = [1/len(cell.neighbors)] * len(cell.neighbors)
                return proportions
            #flat and sink
            else:
                # not needed since sinks wont do flowout routines
                return []


def search(start,path_id,method = "dfs",max_iterations = None,min_flow = None):
    if method == "dfs":
        stack = [start]
    elif method == "bfs":
        stack = deque([start])
    else:
        raise Exception("""Invalid search method. Use "dfs" or "bfs" only.""")
        
    start.explore(path_id)
    
    iterations = 0
    while len(stack)>0:
        
        if method == "dfs":
            node = stack.pop()
        elif method == "bfs":
            node = stack.popleft()

            
        parents = node.parents
        for parent in parents:
            if not parent.is_explored:
                if (min_flow is None) or (parent.accum >= min_flow):
                    stack.append(parent)
                    parent.explore(path_id)
                else:
                    parent.explore(None)
                
        if not max_iterations is None:    
            if max_iterations and (iterations > max_iterations):
                break
        
        iterations += 1
                
    
    
def strahler_search_downstream(start,river,method = "dfs",max_iterations = None):
    if method == "dfs":
        stack = [start]
    elif method == "bfs":
        stack = deque([start])
    else:
        raise Exception("""Invalid search method. Use "dfs" or "bfs" only.""")
        
    explored = set([])
    iterations = 0
    while len(stack)>0:
        
        if method == "dfs":
            parent = stack.pop()
        elif method == "bfs":
            parent = stack.popleft()

        children = parent.neighbors
        for child in children:
            if not child in explored and child in river:
                stack.append(child)
                explored.add(child)
                parent.propagate_to(child)
                
        if not max_iterations is None:    
            if max_iterations and (iterations > max_iterations):
                break
        
        iterations += 1  
    
    
    
    
    
def order_vertices(vertices):
    # Calculate the centroid of the vertices
    centroid = np.mean(vertices, axis=0)

    # Calculate the angles between each vertex and the centroid
    angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])

    # Sort the vertices based on the angles
    sorted_indices = np.argsort(angles)
    ordered_vertices = vertices[sorted_indices]

    # Close the polygon
    ordered_vertices = np.vstack([ordered_vertices, ordered_vertices[0]])
    return ordered_vertices
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
