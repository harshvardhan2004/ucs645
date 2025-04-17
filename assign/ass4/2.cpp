#include <iostream>
#include <omp.h>
#include <vector>

void merge(std::vector<int>& arr, int left, int mid, int right) {
    std::vector<int> leftSub(arr.begin() + left, arr.begin() + mid + 1);
    std::vector<int> rightSub(arr.begin() + mid + 1, arr.begin() + right + 1);

    int i = 0, j = 0, k = left;
    while (i < leftSub.size() && j < rightSub.size()) {
        arr[k++] = (leftSub[i] <= rightSub[j]) ? leftSub[i++] : rightSub[j++];
    }
    while (i < leftSub.size()) arr[k++] = leftSub[i++];
    while (j < rightSub.size()) arr[k++] = rightSub[j++];
}

void parallelMergeSort(std::vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        if (depth < 4) { 
            #pragma omp parallel sections
            {
                #pragma omp section
                parallelMergeSort(arr, left, mid, depth + 1);
                #pragma omp section
                parallelMergeSort(arr, mid + 1, right, depth + 1);
            }
        } else {
            parallelMergeSort(arr, left, mid, depth + 1);
            parallelMergeSort(arr, mid + 1, right, depth + 1);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    std::vector<int> data(1000);
    
    for (int i = 0; i < 1000; ++i) data[i] = rand() % 10000;

    double start = omp_get_wtime();
    parallelMergeSort(data, 0, data.size() - 1);
    double end = omp_get_wtime();

    std::cout << "Pipelined Parallel Merge Sort Time: " << (end - start) << " seconds\n";
    return 0;
}