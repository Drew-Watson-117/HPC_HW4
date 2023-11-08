#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void printIntArray(int* a, int size);
void printFloatArray(float* a, int size);

int main(int argc, char const* argv[])
{       
        int myRank, threadCount, localDataCount, binCount, dataCount;
        float minMeas, maxMeas;
        float* binMaxes;
        int* globalBinCounts;
        float *localData = NULL;
        float *data = NULL;
        int* sendCounts = NULL;

        MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    
        
        if (myRank == 0) {
            if (argc == 5) {

                binCount = atoi(argv[1]);
                minMeas = atof(argv[2]);
                maxMeas = atof(argv[3]);
                dataCount = atoi(argv[4]);

                globalBinCounts = (int*)malloc(binCount*sizeof(int));
                sendCounts = (int*)malloc(threadCount*sizeof(int));
                binMaxes = (float*)malloc(binCount*sizeof(float));

                // Find bin maxes
                for (int i = 0; i < binCount; i++) {
                    binMaxes[i] = minMeas + i * (maxMeas - minMeas) / binCount;
                }

                // Generate Random data
                srand(100);
                float* data = (float*)malloc(dataCount*sizeof(float));
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
                MPI_Bcast(binMaxes, dataCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(globalBinCounts,binCount,MPI_INT,0,MPI_COMM_WORLD);

                // Send sub arrays of data to different threads

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
                MPI_Bcast(sendCounts,threadCount,MPI_INT,0,MPI_COMM_WORLD);

                // Send localDataCount
                // MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
                // Send localData
                MPI_Scatterv(data,sendCounts,0,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); 
                // Set own localDataCount and localData
                memcpy(localData,data,sendCounts[0]*sizeof(float));
                free(data);
                localDataCount = sendCounts[0];
            }
            else {
                printf("Error: Invalid number of command line arguments specified");
                return 1;
            }  
        }
        else {
            // Collect localDataCount and localData from rank 0

            // MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            localDataCount = sendCounts[myRank];
            localData = (float*)malloc(localDataCount*sizeof(float));
            MPI_Scatterv(NULL,sendCounts,0,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); 
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

        // Rank 0 prints the results
        if (myRank == 0)
        {
            printf("Bin Maxes: ");
            printFloatArray(binMaxes, binCount);
            printf("Bin Counts: ");
            printIntArray(globalBinCounts, binCount);
        }

        free(binMaxes), free(globalBinCounts), free(localData), free(sendCounts), free(localBinCounts);

        MPI_Finalize();  

    return 0;
}

void printIntArray(int* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d , ",a[i]);
    }
    printf("\n");
}

void printFloatArray(float* a, int size) {
    for (int i = 0; i < size; i++) {
        printf("%f , ",a[i]);
    }
    printf("\n");
}

