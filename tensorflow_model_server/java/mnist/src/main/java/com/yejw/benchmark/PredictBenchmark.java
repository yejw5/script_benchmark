package com.yejw.benchmark;

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

import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * PredictBenchmark
 *
 */
public class PredictBenchmark
{
    private static final Logger logger = Logger.getLogger(PredictBenchmark.class.getName());
    private final ManagedChannel channel;
    private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

    public PredictBenchmark(String host, int port)
    {
        channel = NettyChannelBuilder.forAddress(host, port)
                .usePlaintext(true)
                .maxMessageSize(100 * 1024 * 1024)
                .build();
        blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    public static void main( String[] args )
    {
        System.out.println("Start the predict client");

        String host = "127.0.0.1";
        int port = 9000;
        String modelName = "mnist";
        long modelVersion = 1;

        // Parse command-line arguments
        if (args.length == 4) {
            host = args[0];
            port = Integer.parseInt(args[1]);
            modelName = args[2];
            modelVersion = Long.parseLong(args[3]);
        }

        // Run predict client to send request
        PredictBenchmark client = new PredictBenchmark(host, port);

        try {
            Predict.PredictRequest request = client.getRequest(modelName, modelVersion);
            client.predict(request);
        } catch (Exception e) {
            System.out.println(e);
        } finally {
            try {
                client.shutdown();
            } catch (Exception e) {
                System.out.println(e);
            }
        }

        System.out.println("End of predict client");
    }

    public Predict.PredictRequest getRequest(String modelName, long modelVersion)
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
        com.google.protobuf.Int64Value version = com.google.protobuf.Int64Value.newBuilder().setValue(modelVersion).build();
        Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setSignatureName("predict_images").setVersion(version).build();
        Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec).putInputs("images", featuresTensorProto).build();

        return request;
    }

    public void predict(Predict.PredictRequest request)
    {
        // Request gRPC server
        Predict.PredictResponse response;
        try {
            response = blockingStub.withDeadlineAfter(10, TimeUnit.SECONDS).predict(request);
            java.util.Map<java.lang.String, org.tensorflow.framework.TensorProto> outputs = response.getOutputs();
            for (java.util.Map.Entry<java.lang.String, org.tensorflow.framework.TensorProto> entry : outputs.entrySet()) {
                System.out.println("Response with the key: " + entry.getKey() + ", value: " + entry.getValue());
            }
        } catch (StatusRuntimeException e) {
            logger.log(Level.WARNING, "RPC failed: {0}", e.getStatus());
            return;
        }
    }
}
