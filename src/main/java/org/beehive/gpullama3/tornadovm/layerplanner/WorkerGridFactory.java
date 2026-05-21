package org.beehive.gpullama3.tornadovm.layerplanner;

import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;

public class WorkerGridFactory {
    private static final int DEFAULT_WORK_GROUP_SIZE = 32;

    /**
     * RMS Norm worker: parallel reduction across dimension // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1]) // CUDA equivalent:
     * kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
     */
    public static WorkerGrid createRmsNormWorker(int dim, int localSize) {
        WorkerGrid worker = new WorkerGrid1D(dim);
        worker.setGlobalWork(dim, 1, 1);
        worker.setLocalWork(localSize, 1, 1);
        return worker;
    }

    // Single worker for tasks running with a single thread
    // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[1,1,1], localWorkSize=[1,1,1])
    // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
    public static WorkerGrid createSingleWorker() {
        WorkerGrid worker = new WorkerGrid1D(1);
        worker.setGlobalWork(1, 1, 1);
        worker.setLocalWork(1, 1, 1);
        return worker;
    }

    /**
     * QKV matmul worker: combined projection output
     */
    public static WorkerGrid createQkvMatmulWorker(int opSize) {
        int global = opSize * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    public static WorkerGrid genericWorker(int globalWorkSize, int localWorkSize) {
        WorkerGrid worker = new WorkerGrid1D(globalWorkSize);
        worker.setLocalWork(localWorkSize, 1, 1);
        return worker;
    }

    /**
     * RoPE worker: 2D grid for position encoding
     */
    public static WorkerGrid createRoPEWorker(int numberOfHeads, int headSize) {
        int ic = headSize / 2;
        WorkerGrid worker = new WorkerGrid2D(numberOfHeads, ic);
        worker.setGlobalWork(numberOfHeads, ic, 1);
        worker.setLocalWork(8, 1, 1);
        return worker;
    }

    /**
     * Attention worker: compute all heads in parallel
     */
    public static WorkerGrid createAttentionWorker(int numberOfHeads, int headSize) {
        int optimalLocalSize = findOptimalLocalSize(headSize);
        WorkerGrid worker = new WorkerGrid1D(numberOfHeads);
        worker.setGlobalWork(numberOfHeads * optimalLocalSize, 1, 1);
        worker.setLocalWork(optimalLocalSize, 1, 1);
        return worker;
    }

    /**
     * FFN gate+up worker: combined projection
     */
    public static WorkerGrid createGateUpWorker(int hiddenDim) {
        int global = (2 * hiddenDim) * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    /**
     * FFN down worker: final projection
     */
    public static WorkerGrid createDownWorker(int dim) {
        int global = dim * DEFAULT_WORK_GROUP_SIZE;
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(DEFAULT_WORK_GROUP_SIZE, 1, 1);
        return worker;
    }

    public static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) {
                    optimal = s;
                    break;
                }
            }
        }
        return optimal;
    }

}
