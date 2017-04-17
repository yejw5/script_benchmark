package com.yejw.benchmark;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.Uninterruptibles;

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
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
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


    public static void main( String[] args )
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

        for ( int i = 0; i < concurrency; ++i ) {
            Thread t = new RequestsThread(host, port, testNum);
            // Run predict client to send request
            t.start();
        }

        System.out.println("End of predict client");
    }

}

class RequestsThread extends Thread {
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceFutureStub stub;
    private final int testNum;

    private void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public RequestsThread(String host, int port, int testNum) {
        channel = NettyChannelBuilder.forAddress(host, port)
            .usePlaintext(true)
            .maxMessageSize(100 * 1024 * 1024)
            .build();
        stub = PredictionServiceGrpc.newFutureStub(channel);
        this.testNum = testNum;
    }
    
    private Predict.PredictRequest getRequest()
    {
        // Generate features TensorProto
        float[][] featuresTensorData = new float[1][784];
        for (int i = 0; i < 1; ++i)
            for (int j = 0; j < 784; ++j)
            {
                featuresTensorData[i][j] = 0.5f;
            }

        TensorProto.Builder featuresTensorBuilder = TensorProto.newBuilder();

        for (int i = 0; i < featuresTensorData.length; ++i) {
            for (int j = 0; j < featuresTensorData[i].length; ++j) {
                featuresTensorBuilder.addFloatVal(featuresTensorData[i][j]);
            }
        }

        TensorShapeProto.Dim featuresDim1 = TensorShapeProto.Dim.newBuilder().setSize(1).build();
        TensorShapeProto.Dim featuresDim2 = TensorShapeProto.Dim.newBuilder().setSize(784).build();
        TensorShapeProto featuresShape = TensorShapeProto.newBuilder().addDim(featuresDim1).addDim(featuresDim2).build();
        featuresTensorBuilder.setDtype(org.tensorflow.framework.DataType.DT_FLOAT).setTensorShape(featuresShape);
        TensorProto featuresTensorProto = featuresTensorBuilder.build();

        // Generate gRPC request
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(1).build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName("mnist").setSignatureName("predict_images").setVersion(version).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("images", featuresTensorProto).build();

        return request;
    }

    @Override
    public void run()
    {
        Predict.PredictRequest request = getRequest();

        final CountDownLatch latch = new CountDownLatch(testNum);
        ListenableFuture<Predict.PredictResponse> response;

        for ( int i = 0; i < testNum; ++i ){
            // Request gRPC server
            response = stub.predict(request);

            Futures.addCallback(
                    response,
                    new FutureCallback<Predict.PredictResponse>() {
                        @Override
                        public void onSuccess(@Nullable Predict.PredictResponse result) {
                            /*
                            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = result.getOutputs();
                            for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                                System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
                            }
                            */
                            latch.countDown();
                        }
                        @Override
                        public void onFailure(Throwable t) {
                            latch.countDown();
                        }
                    },
                    directExecutor());
        }

        try {
            latch.await();
            shutdown();
        } catch(Exception e) {
            System.out.println(e);
        }
    }
}
