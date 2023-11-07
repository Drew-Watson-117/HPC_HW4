#include <iostream>
#include <cstring>
#include <mpi.h>
#include <vector>

void printArray(int* a, int size);
void printArray(float* a, int size);

int main(int argc, char const* argv[])
{
    if (argc == 5)
    {
        int binCount = atoi(argv[1]);
        float minMeas = atof(argv[2]);
        float maxMeas = atof(argv[3]);
        int dataCount = atoi(argv[4]);
        int myRank;
        int threadCount;
        float* localData;
        int localDataCount;
        std::vector<float> maxes;
        // Find bin maxes
        for (int i = 0; i < binCount; i++) {
            maxes.push_back(minMeas + i * (maxMeas - minMeas) / binCount);
        }
        float* binMaxes = maxes.data();
        // Initialize globalBinCounts
        std::vector<int> binCounts(binCount,0);
        int* globalBinCounts = binCounts.data();

        MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        

        if (myRank == 0) {

            srand(100);

            std::vector<float> temp(dataCount, 0);
            float* data = temp.data();

            // Load data with random float values
            for (int i = 0; i < dataCount; i++)
            {
                float randomFloat = (maxMeas - minMeas) * (float)(rand()) / (float)(RAND_MAX)+minMeas; //Guarantees randomFloat to be in [minMeas,maxMeas]
                data[i] = randomFloat;
            }   
            // Send sub arrays of data to different threads
            //int sendCounts[threadCount];
            std::vector<int> send_counts(threadCount, 0);
            int* sendCounts = send_counts.data();

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
            //MPI_Scatterv(data,sendCounts,0,MPI_FLOAT,localData,sendCounts[myRank],MPI_FLOAT,0,MPI_COMM_WORLD); 
            MPI_Scatterv(data, sendCounts, 0, MPI_FLOAT, NULL, NULL, MPI_FLOAT, 0, MPI_COMM_WORLD);
            localDataCount = sendCounts[0];
            std::vector<float> local_data(localDataCount, 0);
            float* localData = local_data.data();
            std::memcpy(localData, data, sendCounts[0] * sizeof(float));
        }
        else {
            MPI_Scatter(NULL, NULL, MPI_INT, &localDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<float> local_data(localDataCount, 0);
            float* localData = local_data.data();
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

