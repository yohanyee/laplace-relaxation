from pyminc.volumes.factory import volumeFromFile, volumeLikeFile, volumeFromData
from pyminc.volumes.volumes import mincVolume
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
import numpy as np

# %% MINC-SPECIFIC FUNCTIONS
# Functions that work on mincVolumes

def crop_to_solution(filename, volume_to_crop,
                     atlas, solution_labels, padding=1,
                     write=True, close=False):
    """
    Crop volume to solution, with optional padding.

    Often, Laplace's equation must be solved in a region that is far smaller than the entire image.
    To avoid the overhead of handling an entire image (the solver loops through each voxel), this function finds
    the bounds around the solution region. Extra padding can be optionally specified to allow for boundary voxels
    and further dilations. It is important to specify a padding of at least 1, to allow for boundary
    definitions; if dilate_laplace() is expected to be called, then it is highly recommended to specify a padding
    of 2 + extend, where extend is the expected number of voxels to expand the solution region.

    Parameters
    ----------
    filname : str
        A path to the new MINC volume.
    volume_to_crop : mincVolume
        Input volume to crop, based on regional definitions in atlas. volume_to_crop.data should be the same size
        as atlas.
    atlas : array_like
        Atlas array, containing labels (solution_labels) corresponding to the solution region. Should be the same
        size as volume_to_crop.data
    solution_labels : list of ints
        List of integer labels corresponding to solution region.
    padding : integer, optional
        Size of padding around cropped region. It is important that this should be at least 1. The default is 1.
    write : bool, optional
        Should the mincVolume be written to disk? Default is True.
    close : bool, optional
        Should the mincVolume be closed? Default is False.

    Returns
    -------
    cropped_volume : mincVolume
        A mincVolume that is a cropped version of volume_to_crop.
    """

    # Get starts
    cropped_starts = volume_to_crop.getStarts()
    separations = volume_to_crop.getSeparations()

    # Find voxels to crop around
    solution_mask = np.isin(atlas, solution_labels)

    # Find how far from the edge it is
    edge_distance = [(
        min(np.where(solution_mask)[i]),
        solution_mask.shape[i] - max(np.where(solution_mask)[i]) - 1
        ) for i in range(3)]

    # How much padding is required?
    padding_distance = [(padding - edge_distance[i][0] + 1, padding - edge_distance[i][1] + 1) for i in range(3)]
    padding_required = [(max(0, padding_distance[i][0]), max(0, padding_distance[i][1])) for i in range(3)]

    # Pad and fix starts
    padded_solution_mask = np.pad(solution_mask, pad_width=padding_required, mode='constant')
    padded_volume_to_crop = np.pad(volume_to_crop.data, pad_width=padding_required, mode='constant')
    cropped_starts = [cropped_starts[i] - separations[i]*padding_required[i][0] for i in range(3)]

    # Find area to crop within padded volume
    working_mask = binary_dilation(padded_solution_mask,
                                   structure=generate_binary_structure(3, 3),
                                   iterations=padding)

    # Bounding box and associated starts
    working_bounds = tuple(slice(min(a), max(a)+1, 1) for a in np.where(working_mask))
    cropped_starts = [cropped_starts[i] + separations[i]*min(np.where(working_mask)[i]) for i in range(3)]

    # Crop data
    cropped_data = padded_volume_to_crop[working_bounds]

    # Cropped volume
    cropped_volume = volumeFromData(outputFilename=filename, data=cropped_data,
                                    dimnames=volume_to_crop.getDimensionNames(),
                                    starts=cropped_starts,
                                    steps=volume_to_crop.getSeparations(),
                                    volumeType=volume_to_crop.volumeType,
                                    dtype=volume_to_crop.dtype,
                                    labels=volume_to_crop.labels,
                                    x_dir_cosines=volume_to_crop.get_direction_cosines('xspace'),
                                    y_dir_cosines=volume_to_crop.get_direction_cosines('yspace'),
                                    z_dir_cosines=volume_to_crop.get_direction_cosines('zspace'))

    # Set dimnames and starts
    cropped_volume.starts = cropped_starts
    cropped_volume.dimnames = volume_to_crop.getDimensionNames()

    # Finish
    if write:
        cropped_volume.writeFile()

    if close:
        cropped_volume.closeVolume()

    # Return
    return(cropped_volume)


def array_to_mincvolume(filename, array, like,
                        volumeType=None, dtype=None, labels=None,
                        write=True, close=False):
    """
    Create a mincVolume from a data array.

    Create a mincVolume from a data array, using coordinate system information from another volume.

    Parameters
    ----------
    filname : str
        A path to the new MINC volume.
    array : array_like
        Input array to convert to mincVolume.
    like : mincVolume or str
        Either an existing mincVolume object, or a path to one on disk.
    volumeType : str, optional
        MINC type. The default is None.
        If no value is given (default), then volumeType will be set as ushort if the dtype
        is a subtype of np.integer, otherwise volumeType will be set as double.
    dtype : np.dtype, optional
        Datatype for the mincVolume data array. The default is None.
        If no value is given (default), the dtype of array is used.
    labels : bool, optional
        Does the output mincVolume represent integer labels? The default is None.
        If no value is given (default), then labels will be set as True if the dtype
        is a subtype of np.integer, otherwise labels will be set as False.
    write : bool, optional
        Should the mincVolume be written to disk? Default is True.
    close : bool, optional
        Should the mincVolume be closed? Default is False.

    Returns
    -------
    outvol : mincVolume
        An object of mincVolume type.

    """
    if dtype is None:
        dtype = array.dtype
    if labels is None:
        if np.issubdtype(array.dtype, np.integer):
            labels = True
        else:
            labels = False
    if volumeType is None:
        if np.issubdtype(array.dtype, np.integer):
            volumeType='ushort'
        else:
            volumeType='double'
    if like.__class__ == mincVolume:
        outvol = volumeFromData(outputFilename=filename,
                                data=array,
                                dimnames=like.getDimensionNames(),
                                starts=like.getStarts(),
                                steps=like.getSeparations(),
                                volumeType=volumeType,
                                dtype=dtype,
                                labels=labels,
                                x_dir_cosines=[i for i in like._x_direction_cosines],
                                y_dir_cosines=[i for i in like._y_direction_cosines],
                                z_dir_cosines=[i for i in like._z_direction_cosines],
                                )

        # Set dimnames and starts
        outvol.starts = like.getStarts()
        outvol.dimnames = like.getDimensionNames()

    else:
        outvol = volumeLikeFile(likeFilename=like, outputFilename=filename,
                                dtype=dtype, volumeType=volumeType, labels=labels)
        outvol.data = array



    # Finish
    if write:
        outvol.writeFile()

    if close:
        outvol.closeVolume()

    return(outvol)

