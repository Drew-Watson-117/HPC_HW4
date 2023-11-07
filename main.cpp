#include <iostream>
#include <cstring>
#include <mpi.h>

void printArray(int* a, int size);
void printArray(float* a, int size);

int main(int argc, char const* argv[])
{
    if (argc == 5)
    {
        
        int myRank;
        int threadCount;
        int binCount;
        float minMeas;
        float maxMeas;
        int dataCount;
        float* localData;
        int localDataCount;
        float* binMaxes;
        int* globalBinCounts;
        MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        

        if (myRank == 0) {
            binCount = atoi(argv[1]);
            minMeas = atof(argv[2]);
            maxMeas = atof(argv[3]);
            dataCount = atoi(argv[4]);
            // Find bin maxes
            binMaxes[binCount];
            for (int i = 0; i < binCount; i++) {
                binMaxes[i] = minMeas + i * (maxMeas - minMeas) / binCount;
            }
            MPI_Bcast(&binCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&minMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&maxMeas, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&dataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(binMaxes, dataCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
            srand(100);
            //float data[dataCount];
            float* data;

            // Load data with random float values
            for (int i = 0; i < dataCount; i++)
            {
                float randomFloat = (maxMeas - minMeas) * (float)(rand()) / (float)(RAND_MAX)+minMeas; //Guarantees randomFloat to be in [minMeas,maxMeas]
                data[i] = randomFloat;
            }   
            // Send sub arrays of data to different threads
            //int sendCounts[threadCount];
            int* sendCounts;

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
            // Send local data count
            MPI_Scatter(sendCounts, 1, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            // Send local data
            MPI_Scatterv(data,sendCounts,0,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); 
            std::memcpy(localData,data,sendCounts[0]*sizeof(float));
            localDataCount = sendCounts[0];
        }
        else {
            MPI_Scatter(NULL, NULL, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, localData, localDataCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        
        // Compute local histogram

        //int localBinCounts[binCount];
        int* localBinCounts;
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
        std::cout << "Bin Maxes: ";
        printArray(binMaxes, binCount);
        std::cout << "Bin Counts: ";
        printArray(globalBinCounts, binCount);
    }

    return 0;
}

void printArray(int* a, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << ", ";
    }
    std::cout << std::endl;
}
void printArray(float* a, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << a[i] << ", ";
    }
    std::cout << std::endl;
}

