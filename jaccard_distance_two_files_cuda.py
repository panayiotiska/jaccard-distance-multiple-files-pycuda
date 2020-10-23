import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

N = 256

mod = SourceModule("""
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define N 256

    __global__ void BitmappingOnDevice(int  *Bitmap, char  *input)
    {
        int i; 
        int total_threads = gridDim.x * blockDim.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        for (i=idx; input[i]; i+=total_threads) {
            Bitmap[input[i]] = 1;
	    }
    }

    __global__ void IntersectUnionOnDevice(int *IntersectUnion, int *Bitmap1_2)
    {
        int i; 
        int total_threads = gridDim.x * blockDim.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        for(i=idx; i<N; i+=total_threads){
            atomicAdd(  &(IntersectUnion[0]), Bitmap1_2[i] && Bitmap1_2[i+N] );
            atomicAdd(  &(IntersectUnion[1]), Bitmap1_2[i] || Bitmap1_2[i+N] );
	    }
    }
""")

# Read Files
with open('data.txt', 'r') as file:
    buff_1 = file.read().replace('\n', '')
with open('data2.txt', 'r') as file:
    buff_2 = file.read().replace('\n', '')

# Convert the string buffers to ascii values
ascii_array_1 = np.fromiter(buff_1, dtype='c').view(np.int8)
ascii_array_2 = np.fromiter(buff_2, dtype='c').view(np.int8)

# An array to return the results with the same size as the buffer
Bitmap1 = np.zeros(N, dtype=int)
Bitmap2 = np.zeros(N, dtype=int)

# An array to store the calculated Intersect and Union
IntersectUnion = np.zeros(2, dtype=int)

# Call the cuda functions
function_bitmap = mod.get_function("BitmappingOnDevice")
function_bitmap(
    drv.Out(Bitmap1), drv.In(ascii_array_1),
    block = (len(ascii_array_1),1,1), grid = (1,1)
) 

function_bitmap(
    drv.Out(Bitmap2), drv.In(ascii_array_2),
    block = (len(ascii_array_2),1,1), grid = (1,1)
) 

# Concatenate the two bitmaps into one
Bitmap1_2 = np.concatenate((Bitmap1, Bitmap2))

functionInUn = mod.get_function("IntersectUnionOnDevice")
functionInUn(
    drv.Out(IntersectUnion), drv.In(Bitmap1_2),
    block = (N*2,1,1), grid = (1,1)
)

# Print the Intersect and Union
print("Intersect = ",IntersectUnion[0],"\nUnion = ",IntersectUnion[1])

# Calculate and print Jaccard distance
Jaccard = IntersectUnion[0]/IntersectUnion[1]*100.0 #Jaccard = intersection over union
print("Jaccard distance between the text of the two files is : ", Jaccard);