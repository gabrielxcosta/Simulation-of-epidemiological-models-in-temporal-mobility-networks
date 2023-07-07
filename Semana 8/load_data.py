import time
import metaSIR as mSIR

start_time = time.time()

folder_name = 'networks/'
mSIR.get_OD_matrices(folder_name)

print("--- %s seconds ---" % (time.time() - start_time))