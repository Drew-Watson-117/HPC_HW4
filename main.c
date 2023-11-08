#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void printIntArray(int* a, int size);
void printFloatArray(float* a, int size);
void serialHistogram(float* data,int binCount, float minMeas, float maxMeas, int dataCount);

int main(int argc, char const* argv[])
{       
        int myRank, threadCount, localDataCount, binCount, dataCount;
        float minMeas, maxMeas;
        float* binMaxes;
        int* globalBinCounts;
        float *localData = NULL;
        float *data = NULL;
        int* sendCounts = NULL;
        int* displs = NULL;

        MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    
        
        if (myRank == 0) {
            if (argc == 5) {
                binCount = atoi(argv[1]);
                minMeas = atof(argv[2]);
                maxMeas = atof(argv[3]);
                dataCount = atoi(argv[4]);

                globalBinCounts = malloc(binCount*sizeof(int));
                for (int i = 0; i < binCount; i++) {
                    globalBinCounts[i] = 0;
                }

                sendCounts = malloc(threadCount*sizeof(int));
                binMaxes = (float*)malloc(binCount*sizeof(float));

                // Find bin maxes
                for (int i = 0; i < binCount; i++) {
                    binMaxes[i] = minMeas + (i+1) * (maxMeas - minMeas) / binCount;
                }

                // Generate Random data
                srand(100);
                data = (float*)malloc(dataCount*sizeof(float));
                // Load data with random float values
                for (int i = 0; i < dataCount; i++)
                {
                    float randomFloat = (maxMeas - minMeas) * (float)(rand()) / (float)(RAND_MAX)+minMeas; //Guarantees randomFloat to be in [minMeas,maxMeas]
                    data[i] = randomFloat;
                }
                // Broadcast collected data and bin maxes to other threads
                MPI_Bcast(&binCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&minMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&maxMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&dataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(binMaxes, binCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(globalBinCounts,binCount,MPI_INT,0,MPI_COMM_WORLD);

                // Send sub arrays of data to different threads
                int* displs = malloc(threadCount*sizeof(int));

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
                    if (rank == 0) {
                        displs[0] = 0;
                    }
                    else {
                        displs[rank] = displs[rank-1] + sendCounts[rank-1];
                    }
                } 
                // Send localDataCount
                MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
                // Send localData
                // MPI_Scatterv(data,sendCounts,displs,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); // Segmentation Fault due to sendCounts[myRank]
                MPI_Scatterv(data,sendCounts,displs,MPI_FLOAT,localData,0,MPI_FLOAT,0,MPI_COMM_WORLD);
                // Set own localDataCount and localData
                localDataCount = sendCounts[0];
                localData = malloc(localDataCount*sizeof(float));
                memcpy(localData,data,localDataCount*sizeof(float));
            }
            else {
                printf("Error: Invalid number of command line arguments specified");
                return 1;
            }  
        }
        else {
            // Collect localDataCount and localData from rank 0
            MPI_Bcast(&binCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&minMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&maxMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            binMaxes = malloc(binCount*sizeof(float));
            globalBinCounts = malloc(binCount*sizeof(float));
            MPI_Bcast(binMaxes, binCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(globalBinCounts,binCount,MPI_INT,0,MPI_COMM_WORLD);

            MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            localData = malloc(localDataCount*sizeof(float));
            MPI_Scatterv(NULL,sendCounts,displs,MPI_FLOAT,localData,localDataCount,MPI_FLOAT,0,MPI_COMM_WORLD);
        }
        
        // Compute local histogram
        int* localBinCounts = malloc(binCount*sizeof(int));
        for (int i = 0; i < binCount; i++) {
            localBinCounts[i] = 0;
        }
        // Bin the data
        for (int i = 0; i < localDataCount; i++) {
            float dataPoint = localData[i];
            int k = 0;
            int flag = 0;
            while (k < binCount && flag == 0) {
                if (dataPoint <= binMaxes[k]) {
                    localBinCounts[k]++;
                    flag = 1;
                }
                k++;
            }
        }

        // Wait for all local sums to finish
        MPI_Barrier(MPI_COMM_WORLD);

        // Use MPI_Reduce to combine sums

        MPI_Reduce(localBinCounts,globalBinCounts,binCount,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

        // Rank 0 prints the results
        if (myRank == 0)
        {
            printf("Data: ");
            printFloatArray(data,dataCount);
            printf("===== Parallel Histogram =====\n");
            printf("Bin Maxes: ");
            printFloatArray(binMaxes, binCount);
            printf("Bin Counts: ");
            printIntArray(globalBinCounts, binCount);
            serialHistogram(data,binCount,minMeas,maxMeas,dataCount);
            
        }
        free(binMaxes), free(localData), free(sendCounts), free(displs);
        free(localBinCounts);
        free(data);

        MPI_Finalize();  

    return 0;
}

// Function to print the first *size* elements of array a
void printIntArray(int* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ",a[i]);
    }
    printf("\n");
}

// Function to print the first *size* elements of array a
void printFloatArray(float* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f ",a[i]);
    }
    printf("\n");
}

// Serial implementation of histogram computation
void serialHistogram(float* data,int binCount, float minMeas, float maxMeas, int dataCount) {
    float* binMaxes = malloc(binCount*sizeof(float));
    int* binCounts = malloc(binCount*sizeof(int));
    for (int i = 0; i < binCount; i++) {
        binMaxes[i] = minMeas + (i+1) * (maxMeas - minMeas) / binCount;
        binCounts[i] = 0;
    }
    for (int i = 0; i < dataCount; i++) {
        float dataPoint = data[i];
        int k = 0;
        int flag = 0;
        while (k < binCount && flag == 0) {
            if (dataPoint <= binMaxes[k]) {
                binCounts[k]++;
                flag = 1;
            }
            k++;
        }
    }
    printf("===== SERIAL HISTOGRAM =====\n");
    printf("Bin Maxes: ");
    printFloatArray(binMaxes, binCount);
    printf("Bin Counts: ");
    printIntArray(binCounts, binCount);
}