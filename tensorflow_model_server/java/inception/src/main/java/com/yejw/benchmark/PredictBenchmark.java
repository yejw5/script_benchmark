package com.yejw.benchmark;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.protobuf.ByteString;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.NettyChannelBuilder;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.lang.Thread;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * AsyncPredictBenchmark
 *
 */
public class AsyncPredictBenchmark
{
    private static final Logger logger = Logger.getLogger(AsyncPredictBenchmark.class.getName());


    public static void main( String[] args ) throws Exception 
    {
        System.out.println("Start the predict client");

        String host = "127.0.0.1";
        int port = 9000;
        int testNum = 10;
        int concurrency = 10;

        // Parse command-line arguments
        if (args.length == 4) {
            host = args[0];
            port = Integer.parseInt(args[1]);
            testNum = Integer.parseInt(args[2]);
            concurrency = Integer.parseInt(args[3]);
        }

        ExecutorService threadPool = Executors.newCachedThreadPool();
        long start = System.currentTimeMillis();

        List<Future<Long>> taskList = new ArrayList<Future<Long>>();
        for (int i = 0; i < concurrency; i++) {
            Future<Long> result = threadPool.submit(new RequestsThread(host, port, testNum));
            taskList.add(result);
        }

        double avg_cost_time = 0.0;
        while (!taskList.isEmpty()) {
            Future<Long> task = taskList.get(0);
            avg_cost_time += (task.get() / 1000.0);
            taskList.remove(0);
        }
        threadPool.shutdown();
        avg_cost_time /= concurrency;

        long end = System.currentTimeMillis();
        System.out.println("average cost time: " + avg_cost_time + "(s)");
        System.out.println("qps: " + (testNum * concurrency / avg_cost_time));

        System.out.println("End of predict client");
    }

}

class RequestsThread implements Callable {
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceFutureStub stub;
    private final int testNum;

    private void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public RequestsThread(String host, int port, int testNum) {
        channel = NettyChannelBuilder.forAddress(host, port)
            .usePlaintext(true)
            .maxMessageSize(1000 * 1024 * 1024)
            .build();
        stub = PredictionServiceGrpc.newFutureStub(channel);
        this.testNum = testNum;
    }

    private Predict.PredictRequest getRequest()
    {
        // Generate features TensorProto
        TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();

        String imagePath = "./data/1.jpg";

        try {
            InputStream imageStream = new FileInputStream(imagePath);
            ByteString imageData = ByteString.readFrom(imageStream);

            featuresTensorBuilder.addStringVal(imageData);

            imageStream.close();
        } catch (IOException e) {
            System.out.println(e);
            System.exit(1);
        }

        TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).build();
        featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_STRING).setTensorShape(featuresShape);
        TensorProto featuresTensorProto = featuresTensorBuilder.build();

        // Generate gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(1).build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName("inception").setSignatureName("predict_images").setVersion(version).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("images", featuresTensorProto).build();

        return request;
    }

    @Override
    public Long call() throws Exception {
        Predict.PredictRequest request = getRequest();

        long start = System.currentTimeMillis();
        long end;
        for ( int i = 0; i < testNum; ++i ){
            Predict.PredictResponse response;
            try {
                response = blockingStub.withDeadlineAfter(10, TimeUnit.SECONDS).predict(request);
                java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputs();
                System.out.print(".");
                /*
                for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                    System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
                }
                */
            } catch (StatusRuntimeException e) {
                logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            }
        }

        System.out.println("Thread " + Thread.currentThread().getId() + " has send all requests.");

        try {
            end = System.currentTimeMillis();
        } catch(Exception e) {
            System.out.println(e);
            end = System.currentTimeMillis();
        }
        System.out.println("The cost time of thread " + Thread.currentThread().getId() + " is: " + (end - start) + "(ms)");
        return end - start;
    }
}
