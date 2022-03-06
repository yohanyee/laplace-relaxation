from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from numba import njit
import numpy as np

# %% LAPLACE SOLVER FUNCTIONS
# Functions that work on numpy arrays, without any reference to a world coordinate system

def atlas_to_laplace_structure(atlas, solution_labels, dirichlet_labels, neumann_labels):
    """
    Create a structure map of the Laplace problem.

    This function essentially remaps an input atlas to an application-specific one,
    i.e. an atlas with hard-coded labels corresponding to boundary and solution regions.
    This is used by the equation solver to guide which voxels are changed, and is static throughout.

    Parameters
    ----------
    atlas : array_like
        Atlas array.
    solution_labels : list of ints
        List of integer labels corresponding to solution region.
    dirichlet_labels : list of ints
        List of integer labels corresponding to Dirichlet boundary.
    neumann_labels : list of ints
        List of integer labels corresponding to Neumann boundary.

    Returns
    -------
    laplace_structure : array_like
        An array of same size as atlas, with values defining the spatial structure:
            Solution region:    1
            Dirichlet boundary: 2
            Neumann boundary:   3
    """
    laplace_structure = np.zeros_like(atlas)
    laplace_structure[np.isin(atlas, solution_labels)] = 1
    laplace_structure[np.isin(atlas, dirichlet_labels)] = 2
    laplace_structure[np.isin(atlas, neumann_labels)] = 3
    return(laplace_structure)


def atlas_to_laplace_grid(atlas, solution_labels, dirichlet_labels, neumann_labels, remap_labels=None):
    """
    Create the Laplace grid.

    The Laplace grid is the actual specification of the Laplace problem. It is a map with Dirichlet boundary
    conditions set (and these will be held fixed by the solver). Neumann boundary voxels and the initial solution
    space start at zero (by default, can be remapped). This grid is the initial state of the solution map, and
    evolves over each iteration towards the relaxed solution.

    Parameters
    ----------
    atlas : array_like
        Atlas array.
    solution_labels : list of ints
        List of integer labels corresponding to solution region.
    dirichlet_labels : list of ints
        List of integer labels corresponding to Dirichlet boundary.
    neumann_labels : list of ints
        List of integer labels corresponding to Neumann boundary.
    remap_labels : dict, optional
        A dict to map atlas labels to new boundary values. Useful if you want to specify non-integer boundaries.
        Each key:value pair maps corresponds to old_labels:new_labels. The default is None.

    Returns
    -------
    laplace_grid : array_like
        An array of same size as atlas, defining the Laplace grid.

    """
    laplace_grid = np.zeros_like(atlas, dtype=np.float64)
    laplace_grid[np.isin(atlas, solution_labels)] = 0
    laplace_grid[np.isin(atlas, dirichlet_labels)] = atlas[np.isin(atlas, dirichlet_labels)]
    laplace_grid[np.isin(atlas, neumann_labels)] = 0
    if remap_labels is not None:
        for old_label in remap_labels:
            new_label = remap_labels[old_label]
            laplace_grid[np.isin(atlas, old_label)] = new_label
    return(laplace_grid)


@njit
def iterate_laplace(laplace_structure, laplace_solution, method='jacobi', w=1):
    """
    One entire relaxation iteration over all voxels.

    One iteration loops over all voxels, and computes updated values based on either the Jacobi method
    (solution updated at the end), or the Gauss-Seidel method (solution updated in place), the latter
    apparently being around two times faster as information is propagated to edges earlier. An overrelaxation
    parameter may also be specified to speed things up. The basic algorithm is:
        - first use the Laplace structure map to check whether the voxel is part of the Dirichlet boundary,
        Neumann boundary, or the solution space (i.e. should the voxel value be held fixed or not?)
        - based on the voxel classification, if the voxel value can change (Neumann or solution), update the
        value either by considering the average value of directional neighbours (Neumann) or all neighbours (solution).

    Parameters
    ----------
    laplace_structure : array_like
        Structure map of the Laplace problem. An atlas with labels 1 (solution region),
        2 (Dirichlet boundary), and 3 (Neumann boundary), zero everywhere else, and nothing else.
        Output of atlas_to_laplace_structure.
    laplace_solution : array_like
        Laplace grid (current state of the solution map). The actual specification of the relaxation
        problem, with Dirichlet boundary conditions specified. Output of atlas_to_laplace_grid.
    method : str, optional
        Either jacobi or gauss_seidel. The default is 'jacobi'.
    w : float, optional
        Overrelaxation parameter. The default is 1.

    Returns
    -------
    laplace_solution, vchange
        A tuple with the new grid solution (array_like, same size as laplace_structure),
        and the average squared change in voxel values.

    """
    # Solution containers
    if method == 'jacobi':
        new_laplace_solution = np.copy(laplace_solution)

    # Error containers
    vnum = 0
    vdiff = 0.

    # Iterate over each axis within padded region
    for x in range(1, laplace_structure.shape[0]-1):
        for y in range(1, laplace_structure.shape[1]-1):
            for z in range(1, laplace_structure.shape[2]-1):
                # If voxel is part of solution region
                if laplace_structure[x, y, z] == 1:
                    # Update solution
                    # New solution is 6-neighbour average
                    # In the solution region, all neighbours considered (part of solution or boundary)
                    s = 0.
                    s += laplace_solution[x-1, y, z]
                    s += laplace_solution[x+1, y, z]
                    s += laplace_solution[x, y-1, z]
                    s += laplace_solution[x, y+1, z]
                    s += laplace_solution[x, y, z-1]
                    s += laplace_solution[x, y, z+1]
                    # Compute update value with overrelaxation
                    old_value = laplace_solution[x, y, z]
                    new_value = s/6.
                    updated_value = w*new_value + (1-w)*old_value
                    # Update
                    if method == 'jacobi':
                        new_laplace_solution[x, y, z] = updated_value
                    else:
                        laplace_solution[x, y, z] = updated_value
                    # Compute change
                    vdiff += (updated_value - old_value)**2
                    vnum += 1
                    continue

                # If voxel is part of Dirichlet boundary
                if laplace_structure[x, y, z] == 2:
                    # Don't do anything
                    continue

                # If voxel is part of Neumann boundary
                if laplace_structure[x, y, z] == 3:
                    # Update Neumann boundary
                    # Gradient along normal should be zero, therefore take average within solution region only
                    g = 0.
                    n = 0
                    if laplace_structure[x-1, y, z] == 1:
                        g += laplace_solution[x-1, y, z]
                        n += 1
                    if laplace_structure[x+1, y, z] == 1:
                        g += laplace_solution[x+1, y, z]
                        n += 1
                    if laplace_structure[x, y-1, z] == 1:
                        g += laplace_solution[x, y-1, z]
                        n += 1
                    if laplace_structure[x, y+1, z] == 1:
                        g += laplace_solution[x, y+1, z]
                        n += 1
                    if laplace_structure[x, y, z-1] == 1:
                        g += laplace_solution[x, y, z-1]
                        n += 1
                    if laplace_structure[x, y, z+1] == 1:
                        g += laplace_solution[x, y, z+1]
                        n += 1
                    # Compute update value with overrelaxation
                    old_value = laplace_solution[x, y, z]
                    if n >= 1:
                        new_value = g/n
                        updated_value = w*new_value + (1-w)*old_value
                    else:
                        updated_value = old_value
                    # Update
                    if method == 'jacobi':
                        new_laplace_solution[x, y, z] = updated_value
                    else:
                        laplace_solution[x, y, z] = updated_value
                    continue

    if method == 'jacobi':
        laplace_solution = new_laplace_solution

    # Accumulate change
    vchange = vdiff/vnum

    return(laplace_solution, vchange)


def solve_laplace(laplace_structure, laplace_grid, method='jacobi', w=1,
                  output_type=None, output_nonsolution_value=-1,
                  convergence=1e-10, max_iters=10000, verbose=True):
    """
    Solve Laplace's equation.

    Iterate using the relaxation method until convergence.

    Parameters
    ----------
    laplace_structure : array_like
        Structure map of the Laplace problem. An atlas with labels 1 (solution region),
        2 (Dirichlet boundary), and 3 (Neumann boundary), zero everywhere else, and nothing else.
        Output of atlas_to_laplace_structure.
    laplace_solution : array_line
        Laplace grid (current state of the solution map). The actual specification of the relaxation
        problem, with Dirichlet boundary conditions specified. Output of atlas_to_laplace_grid.
    method : str, optional
        Either jacobi or gauss_seidel. The default is 'jacobi'.
    w : float, optional
        Overrelaxation parameter. The default is 1.
    output_type : str, optional
        What to output with the solution.
        If None, just output the solution, and set all voxels surrounding the solution region
            (including the Dirichlet and Neumann boundaries) to be a constant (given by output_nonsolution_value).
        If 'neumann', include the relaxed Neumann boundary values with the solution, and set everything else to
            be a constant (given by output_nonsolution_value).
        If 'boundary', 'boundary2', or 'boundary3', then include all boundary voxels (including relaxed Neumann
            boundary voxel values) with the solution; everything else is constant (given by output_nonsolution_value).
            'boundary2' and 'boundary3' use the 18 and 26 neighbour structuring element to determine boundary voxels
            respectively.
        If 'complete', then the solution and relaxed Neumann boundary values are outputed on top of the original
            laplace grid.
        The default is None.
    output_nonsolution_value : float, optional
        At the end, voxels not part of the solution region are masked out by writing over
        with this value. The default is -1.
    convergence : float, optional
        Average squared change in voxel value, below which the algorithm is considered
        to have converged. The default is 1e-10.
    max_iters : int, optional
        Maximum number of iterations. The default is 10000.
    verbose : bool, optional
        Be chatty. The default is True.

    Returns
    -------
    output : array_like
        The solution to Laplace's equation, same array size as input laplace_structure.

    """
    # Initialize solution array
    laplace_structure = np.pad(laplace_structure, 1, mode='constant')
    laplace_solution = np.pad(laplace_grid, 1, mode='constant')

    # Track iterations
    iters = 0

    # Iterate
    while (iters < max_iters):
        # Iterate
        laplace_solution, average_voxel_change_sq = iterate_laplace(laplace_structure,
                                                                    laplace_solution,
                                                                    method=method, w=w)
        # Status
        if verbose:
            print("Iteration: {} (max: {}) | \
                  Current change: {} (target: {})".format(
                  iters, max_iters, average_voxel_change_sq, convergence
                  )
                  )
        # Save

        # Check for convergence
        if (average_voxel_change_sq) <= convergence:
            break
        # Increase iterations
        iters += 1

    # Unpad solution
    laplace_structure = laplace_structure[tuple(slice(1, a-1, 1) for a in laplace_structure.shape)]
    laplace_solution = laplace_solution[tuple(slice(1, a-1, 1) for a in laplace_solution.shape)]

    # Prepare output
    output = np.full(laplace_solution.shape, output_nonsolution_value, dtype=laplace_solution.dtype)
    if output_type is None:
        valid_mask = np.isin(laplace_structure, [1])
    if output_type == 'neumann':
        valid_mask = np.isin(laplace_structure, [1, 3])
    if output_type == 'boundary':
        valid_mask = np.isin(laplace_structure, [1])
        valid_mask = binary_dilation(valid_mask,
                                     structure=generate_binary_structure(3, 1),
                                     iterations=1)
    if output_type == 'boundary2':
        valid_mask = np.isin(laplace_structure, [1])
        valid_mask = binary_dilation(valid_mask,
                                     structure=generate_binary_structure(3, 2),
                                     iterations=1)
    if output_type == 'boundary3':
        valid_mask = np.isin(laplace_structure, [1])
        valid_mask = binary_dilation(valid_mask,
                                     structure=generate_binary_structure(3, 1),
                                     iterations=3)
    if output_type == 'complete':
        valid_mask = np.isin(laplace_structure, [1, 3])

    output[valid_mask] = laplace_solution[valid_mask]
    if output_type == 'complete':
        output[~valid_mask] = laplace_grid[~valid_mask]


    # Done
    if verbose:
        print("Done")

    return(output)

# Dilate laplace: expand laplace solution to outside
def dilate_laplace(laplace_structure, laplace_solution, extend=1, with_boundary=True,
                   method='jacobi', w=1, output_nonsolution_value=-1,
                   convergence=1e-10, max_iters=10000, verbose=True):
    """
    Expand the relaxed solution past the boundaries.

    Expand the relaxed solution within a larger region that extends past the boundaries.
    This is useful when computing paths of steepest descent along the solution, and you want
    the paths to not be affected by the sharp cutoffs that exist at the edges of the solution.
    Expansion is achieved by first dilating a mask corresponding to the original solution region,
    and then solving Laplace's equation again in the dilated region; thus some arguments to this
    function correspond to those of solve_laplace(). Note: dilate_laplace assumes that the input
    array is large enough to allow for dilations; if not, the output of this function may not be
    what you want. It is therefore highly recommended that the input arrays be appropriately padded
    so that they are large enough, for example, by running crop_to_solution at some earlier point
    in the workflow. Despite its name, crop_to_solution can also expand images by padding.


    Parameters related to solution expansion
    ----------
    laplace_structure : array_like
        Structure map of the Laplace problem. An atlas with labels 1 (solution region),
        2 (Dirichlet boundary), and 3 (Neumann boundary), zero everywhere else, and nothing else.
        Output of atlas_to_laplace_structure.
    laplace_solution : array_line
        Laplace grid (current state of the solution map). The actual specification of the relaxation
        problem, with Dirichlet boundary conditions specified. Output of atlas_to_laplace_grid.
    extend : int, optional
        How many voxels to dilate? The default is 1.
    with_boundary : bool, optional
        Include the boundary? The default is True.

    Parameters related to laplace solver
    ----------
    method : str, optional
        Either jacobi or gauss_seidel. The default is 'jacobi'.
    w : float, optional
        Overrelaxation parameter. The default is 1.
    output_nonsolution_value : float, optional
        At the end, voxels not part of the solution region are masked out by writing over
        with this value. The default is -1.
    convergence : float, optional
        Average squared change in voxel value, below which the algorithm is considered
        to have converged. The default is 1e-10.
    max_iters : int, optional
        Maximum number of iterations. The default is 10000.
    verbose : bool, optional
        Be chatty. The default is True.

    Returns
    -------
    laplace_solution_dilated : array_like
        An array (larger than the input) corresponding to the expanded solution.

    """

    # Create new laplace structure specific to this problem
    laplace_solution_mask = laplace_structure == 1
    if with_boundary:
        laplace_solution_mask = binary_dilation(laplace_solution_mask,
                                                structure=generate_binary_structure(3, 1),
                                                iterations=1)
    laplace_solution_mask_extended = binary_dilation(laplace_solution_mask,
                                                     structure=generate_binary_structure(3, 1),
                                                     iterations=extend)
    laplace_solution_mask_extended_boundary = binary_dilation(laplace_solution_mask,
                                                              structure=generate_binary_structure(3, 1),
                                                              iterations=(extend + 1))
    neumann_boundary = laplace_solution_mask_extended_boundary & ~laplace_solution_mask_extended
    dirichlet_boundary = laplace_solution_mask
    solution_region = laplace_solution_mask_extended & ~laplace_solution_mask
    laplace_structure_dilation_problem = np.zeros_like(laplace_structure, dtype=laplace_structure.dtype)
    laplace_structure_dilation_problem[neumann_boundary] = 3
    laplace_structure_dilation_problem[solution_region] = 1
    laplace_structure_dilation_problem[dirichlet_boundary] = 2

    # Create laplace grid specific to this problem
    laplace_grid_dilation_problem = np.zeros_like(laplace_solution, dtype=laplace_solution.dtype)
    laplace_grid_dilation_problem[laplace_solution_mask] = laplace_solution[laplace_solution_mask]

    # Solve laplace
    laplace_solution_dilated = solve_laplace(laplace_structure_dilation_problem,
                                             laplace_grid_dilation_problem, output_type='complete',
                                             method=method, w=w, output_nonsolution_value=output_nonsolution_value,
                                             convergence=convergence, max_iters=max_iters, verbose=verbose)

    # Patch original solution back in
    laplace_solution_dilated[laplace_solution_mask] = laplace_solution[laplace_solution_mask]

    # Return
    return(laplace_solution_dilated)

# Make movie
def relaxation_movie():
    #TODO: implement a procedure to track intermediate steps of the relaxation and output either as
    # a 4D MINC file and/or a movie.
    pass
