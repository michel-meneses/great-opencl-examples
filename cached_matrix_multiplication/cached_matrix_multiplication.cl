/**
 * This kernel function efficiently multiplies two matrices a[M,K] and b[K,N] 
 * by caching submatrices from those input matrices in the device local memory.
 */

__kernel void multiplyMatricesWithCache(__constant int* a,
                                    __constant int* b,
                                    __global int* c,
                                    const int M,
                                    const int N, 
                                    const int K){

    /**
     * Declare the size of each submatrix (it must be 
     * the same work-group size declared in the host code).
     */

    const int SUB_SIZE = 16;
    
    /**
     * Get work-item identifiers.
     */
    
    int colIndex = get_local_id(0);
    int rowIndex = get_local_id(1);
    int globalColIndex = get_global_id(0);
    int globalRowIndex = get_global_id(1);
    int index = (globalRowIndex * N) + globalColIndex;

    /**
     * Create submatrices that will cache the matrices A and B in local memory.
     */

    __local int aSub[SUB_SIZE][SUB_SIZE];
    __local int bSub[SUB_SIZE][SUB_SIZE];

    /**
     * Initialize accumulator register.
     */

    int sum = 0;

    /**
     * Loop over all submatrices.
     */

    const int nSub = K / SUB_SIZE;
    for(int s = 0; s < nSub; s++){

        /**
         * Load submatrices into local memory.
         */

        const int sCol = SUB_SIZE * s + colIndex;
        const int sRow = SUB_SIZE * s + rowIndex;
        aSub[rowIndex][colIndex] = a[globalRowIndex * K + sCol];
        bSub[rowIndex][colIndex] = b[sRow * N + globalColIndex];

        /**
         * Synchronize all work-items in this work-group.
         */

        barrier(CLK_LOCAL_MEM_FENCE);

        /**
         * Perform the computation for a single submatrix.
         */
        
        for(int k = 0; k < SUB_SIZE; k++){
            sum += aSub[rowIndex][k] * bSub[k][colIndex];
        }

        /**
         * Synchronize all work-items in this work-group.
         */

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /**
     * Store the final result in the matrix C.
     */

    c[index] = sum;
}