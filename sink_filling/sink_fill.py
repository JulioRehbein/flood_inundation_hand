import numpy as np
import subprocess
import matplotlib.pyplot as plt
import earthpy.plot as ep
import earthpy.spatial as es

plt.close("all")

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



#%%%
epsilon = 0.0001
executable = "./planchon_darboux.exe"
args = ["dem.npy",str(epsilon),"filled.npy"]

try:
    result = subprocess.run([executable] + args,
                            stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE,
                            text = True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)
        
except FileNotFoundError:
    print("Executable not found.")
    
dem_resolution = 30
dem = np.flip(np.load("dem.npy"),axis = 0)
filled = np.flip(np.load("filled.npy"),axis = 0)



#%%




fig = plt.figure()

az,alt = 270,10


hillshade = es.hillshade(dem,azimuth = az,altitude=alt)
ax = fig.add_subplot(121)
ep.plot_bands(hillshade,cbar = False,ax = ax)

hillshade = es.hillshade(filled,azimuth = az,altitude=alt)
ax = fig.add_subplot(122)
ep.plot_bands(hillshade,cbar = False,ax = ax)

diff = filled-dem
diff[diff == 0] = np.nan

xs,ys = np.meshgrid(range(dem.shape[1]), range(dem.shape[0]))
img = ax.pcolormesh(xs,ys,diff,alpha = 0.7,cmap = "RdBu")
#fig.colorbar(img,ax=ax,orientation = "horizontal")
plt.show()



























