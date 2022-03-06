from pyminc.volumes.factory import volumeFromFile
from laplacerelaxation.minc_interface import crop_to_solution, array_to_mincvolume
from laplacerelaxation.relaxation import atlas_to_laplace_structure, atlas_to_laplace_grid
from laplacerelaxation.relaxation import solve_laplace
from matplotlib import pylab as plt

# Inputs
atlas_file = "laplace_atlas.mnc"
atlas_vol = volumeFromFile(atlas_file, labels=True)
solution_labels = [2]
dirichlet_labels = [0, 1]
neumann_labels = [3]
extend=1
remap_labels = None

if __name__ == '__main__':

    # Crop atlas around solution region
    atlas_cropped = crop_to_solution("laplace_atlas_cropped.mnc", atlas_vol,
                                     atlas=atlas_vol.data, solution_labels=[2], padding=(2+extend),
                                     close=True)
    atlas_original = atlas_vol.data
    atlas = atlas_cropped.data

    # Setup laplace problem
    laplace_structure = atlas_to_laplace_structure(atlas, solution_labels, dirichlet_labels, neumann_labels) # Define structure
    laplace_grid = atlas_to_laplace_grid(atlas, solution_labels, dirichlet_labels, neumann_labels) # Initial state

    # Solve
    laplace_solution = solve_laplace(laplace_structure, laplace_grid, max_iters=1000, method='jacobi', w=1.0)
    laplace_solution.shape

    # Write back
    solution_vol = array_to_mincvolume('laplace_solution_cropped.mnc', laplace_solution, like=atlas_cropped)
    solution_vol.closeVolume()

    # Examine what each image looks like
    #
    #plt.imshow(laplace_structure[:, 50, :])
    #plt.colorbar()
    #
    #plt.imshow(laplace_grid[:, 50, :])
    #plt.colorbar()
    #
    #plt.imshow(laplace_solution[:, 50, :])
    #plt.colorbar()
    #
    #plt.imshow(laplace_dilated[:, 50, :])
    #plt.colorbar()