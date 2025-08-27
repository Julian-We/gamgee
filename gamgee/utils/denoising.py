def get_shared_memory_info(identifier_str:str):
    """
    Get information about the shared memory segment.

    :param identifier_str: Identifier string for the shared memory segment.
    :return: Dictionary containing shared memory information.
    """
    shm_name, shape_raw, dtype = identifier_str.split('-')
    shape = tuple(map(int, shape_raw.split('_')))

    return {
        "name": shm_name,
        "shape": shape,
        "dtype": dtype
    }