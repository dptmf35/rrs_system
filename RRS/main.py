from get_RRS import get_RRS
import time

start = time.time()


rrs = get_RRS()

mem_no = 1234564856 # member numb

print(rrs.get_RRS(mem_no))

print(f"Elapsed time for {mem_no} : {time.time() - start :.3f}")

