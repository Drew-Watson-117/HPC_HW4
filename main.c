#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void printArray(int* a, int size);
void printArray(float* a, int size);

int main(int argc, char const* argv[])
{
    if (argc == 5)
    {
        
        int myRank;
        int threadCount;
        int binCount = atoi(argv[1]);
        float minMeas = atof(argv[2]);
        float maxMeas = atof(argv[3]);
        int dataCount = atoi(argv[4]);
        float* localData;
        int localDataCount;
        float* binMaxes = (float*)malloc(binCount*sizeof(float));
        int* globalBinCounts = (int*)malloc(binCount*sizeof(int));

        // Find bin maxes
        for (int i = 0; i < binCount; i++) {
            binMaxes[i] = minMeas + i * (maxMeas - minMeas) / binCount;
        }

        srand(100);

        float* data = (int*)malloc(dataCount*sizeof(float));
        // Load data with random float values
        for (int i = 0; i < dataCount; i++)
        {
            float randomFloat = (maxMeas - minMeas) * (float)(rand()) / (float)(RAND_MAX)+minMeas; //Guarantees randomFloat to be in [minMeas,maxMeas]
            data[i] = randomFloat;
        }   

        MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        

        if (myRank == 0) {

            // Send sub arrays of data to different threads
            int* sendCounts = (int*)malloc(threadCount*sizeof(int));

            for (int rank = 0; rank < threadCount; rank++)
            {
                // Make sub array
                int localCount;
                if (rank < dataCount % threadCount) {
                    // size is dataCount / threadCount + 1
                    localCount = dataCount / threadCount + 1;
                }
                else {
                    // size is dataCount / threadCount
                    localCount = dataCount / threadCount;
                }
                sendCounts[rank] = localCount;
            } 
            // Send localDataCount
            MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            // Send localData
            MPI_Scatterv(data,sendCounts,0,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); 
            // Set own localDataCount and localData
            memcpy(localData,data,sendCounts[0]*sizeof(float));
            localDataCount = sendCounts[0];
        }
        else {
            // Collect localDataCount and localData from rank 0
            MPI_Scatter(NULL, NULL, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            localData = (float*)malloc(localDataCount*sizeof(float));
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, localData, localDataCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        
        // Compute local histogram

        //int localBinCounts[binCount];
        int* localBinCounts = (int*)malloc(binCount*sizeof(int));
        // Bin the data
        for (int i = 0; i < localDataCount; i++) {
            float dataPoint = localData[i];
            for (int k = 0; k < binCount; k++) {
                if (dataPoint < binMaxes[k]) {
                    localBinCounts[k]++;
                    break;
                }
            }
        }
        // Use MPI_Reduce to combine sums
        for (int i = 0; i < binCount; i++) {
            MPI_Reduce(&(localBinCounts[i]), &(globalBinCounts[i]), 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        MPI_Finalize();

        // Print results serially
        printf("Bin Maxes: ");
        printArray(binMaxes, binCount);
        printf("Bin Counts: ");
        printArray(globalBinCounts, binCount);
    }

    return 0;
}

void printArray(int* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d , ",a[i]);
    }
    printf("\n");
}
void printArray(float* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d , ",a[i]);
    }
    printf("\n");
}

