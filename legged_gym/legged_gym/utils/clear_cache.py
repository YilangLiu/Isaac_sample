from multiprocessing import shared_memory


def clear_cache_name(name):
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
        print(f"Successfully unlinked shared memory. {name}")
    except FileNotFoundError:
        print("Shared memory not already cleared. Proceed to Next one")