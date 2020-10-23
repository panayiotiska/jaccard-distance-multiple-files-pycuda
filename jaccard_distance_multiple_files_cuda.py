import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import itertools
import operator
import glob
import os

N = 256

mod = SourceModule("""
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#define N 256

    __global__ void BitmappingOnDevice(int  *Bitmap, char  *input, int *array_of_lengths, int number_of_files)
    {
        int j,k; 
        int already_read = 0;
        int total_threads = gridDim.x * blockDim.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        for (j=0; j<number_of_files;j++){
            for (k=idx; k<array_of_lengths[j]; k+=total_threads){
                Bitmap[input[k+already_read]+(N*j)] = 1;
            }
            already_read+=array_of_lengths[j];
        }
    }

    __global__ void IntersectUnionOnDevice(int *IntersectUnion, int *Bitmap, int number_of_files)
    {
        int i,j,k,temp = 0;
        int last = ((2^number_of_files)*2)-1; //Pointing to the last digit of the IntersectUnion array
        int total_threads = gridDim.x * blockDim.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        for (j=0; j<number_of_files-1;j++){
            temp += (bool)j; // increase temp by one but not in first repetition
            for (k=j+1; k<number_of_files;k++){
                for(i=idx; i<N; i+=total_threads){
                    atomicAdd(  &(IntersectUnion[j+k-1+temp]), Bitmap[i+(N*j)] && Bitmap[i+(N*k)] );
                    atomicAdd(  &(IntersectUnion[j+k+temp]), Bitmap[i+(N*j)] || Bitmap[i+(N*k)] );
                }
                temp +=1;
            }
        }

        // Calculate the Intersect and Union for the last pair
        j--;
        k--;
        IntersectUnion[last] = 0; //if(threadIdx.x==0) IntersectUnion[last] = 0;
        for(i=idx; i<N; i+=total_threads){
            atomicAdd(  &(IntersectUnion[last-1]), Bitmap[i+(N*j)] && Bitmap[i+(N*k)] );
            atomicAdd(  &(IntersectUnion[last]), Bitmap[i+(N*j)] || Bitmap[i+(N*k)] );
        }
    }

    __global__ void GetMaxJaccardDistanceOnDevice(float  *Jaccard_distances, int  *IntersectUnion, int max)
    {
        int i=0;

        const int j = threadIdx.x;
        i=j*2;
        Jaccard_distances[j] = __fdividef(IntersectUnion[i], IntersectUnion[i+1])*100;
    }

""")

# Read directory path from user
#directory_path = input("insert the path of directory with multiple txt files: ")
directory_path = r'C:\...ADD THE PATH OF THE FOLDER CONTAINING THE FILES'


# Read all files in a directory
os.chdir(directory_path)
myFiles = glob.glob('*.txt')
print("FILE NAMES : ",myFiles)
print("-------------------------------------------------------------")

number_of_files = 0
array_of_lengths = []
buff = []
for file in myFiles:
    with open(file, 'r') as f_input:
        temp_buff = np.fromiter(f_input.read().replace('\n', ''), dtype='c').view(np.int8) # Convert the string buffers to ascii values
        array_of_lengths.append(len(temp_buff))
        buff.append(temp_buff)

buff = list(itertools.chain.from_iterable(buff))
number_of_files = len(array_of_lengths)

print("INPUT IN ASCII : ",buff)
print("ARR OF LENGTHS : ",array_of_lengths)
print("NUMBER OF FILES : ",number_of_files)

# An array to return the results with the same size as the buffer
size_of_Bitmap = N*number_of_files
Bitmap = np.zeros(N*number_of_files, dtype=int)

# An 1D array for storing thw intersection and union for every pair 
# in the form of ((intersect between file1 and file2)(union between file1 and file)(intersect between file1 and file3)(union between file1 and file3)...)
size_of_IntersectUnion = (2^number_of_files)*2
IntersectUnion = np.zeros(size_of_IntersectUnion, dtype=int)

# An array for every pair of files to store their Jaccard distance
size_of_Jaccard_distances = 2^number_of_files
Jaccard_distances = np.zeros(size_of_Jaccard_distances).astype(np.float32)

# Call the cuda functions

# Creates a 1D bit map (array) for all files of size 256*number_of_files, 
# having 1 in the ascii position of each character included in each file (file1 -> 0-255, file2 -> 256-511 etc.) 
function_bitmap = mod.get_function("BitmappingOnDevice")
function_bitmap(
    drv.Out(Bitmap), drv.In(np.array(buff)), drv.In(np.array(array_of_lengths)), (np.int32(number_of_files)),
    block = (len(buff),1,1), grid = (1,1)
)
# with np.printoptions(threshold=np.inf):
#     print("BITMAP : ",Bitmap)

# Usibng the Bitmap calculates the intersection and union of every pair of files and stores them in a 1D array
functionInUn = mod.get_function("IntersectUnionOnDevice")
functionInUn(
    drv.Out(IntersectUnion), drv.In(Bitmap), (np.int32(number_of_files)),
    block = (size_of_Bitmap,1,1), grid = (1,1)
) 
#IntersectUnion = IntersectUnion.astype(float)
print("INTERSETCION AND UNION IN PAIRS OF TWO : ",IntersectUnion)

# Using the IntersectUnion array, calculates the jaccard distances and stores them in a 1D array
functionJaccard = mod.get_function("GetMaxJaccardDistanceOnDevice")
functionJaccard(
    drv.Out(Jaccard_distances), drv.In(IntersectUnion), (np.int32(0)),
    block = (6,1,1), grid = (1,1)
) 
print("JACCARD DISTANCES OF EVERY PAIR: ",Jaccard_distances)

# Find the max jaccard distance
index, value = max(enumerate(Jaccard_distances), key=operator.itemgetter(1))

# Find the two files that have the max jaccard distance
def find_pairs(source):
        result = []
        for p1 in range(len(source)):
                for p2 in range(p1+1,len(source)):
                        result.append([source[p1],source[p2]])
        return result

pairings = find_pairs(myFiles)

# Print results
print("-------------------------------------------------------------")
print("The maximum jaccard distance is ",value," between files ",pairings[index])
