package Bugprediction;

import Bugprediction.Iterators.KFoldIterator;
import Bugprediction.tools.CSVWriter;
import Bugprediction.tools.CSVRecordReader;
import Bugprediction.tools.Evaluator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.util.Log;
import org.joda.time.DateTime;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

/**
 * @author Sébastien Broggi
 */
public class RegressionByClassification {
    private static Logger log = LoggerFactory.getLogger(RegressionByClassification.class);

    public static void main(String[] args) throws  Exception {
        Evaluator evaluator = new Evaluator();
        Date date = new Date();
        SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd HH.mm");
        String dateString = sdfDate.format(date);
        String[] projects = new String[] { "mylynAllMetrics.csv" };
        //String[] projects = new String[] { "mylynAllMetrics.csv", "jdtAllMetrics.csv", "luceneAllMetrics.csv", "pdeAllMetrics.csv", "equinoxAllMetrics.csv" };

        for (String project: projects) {
            int labelIndex = 32;     //32 Features
            int numClasses = 5;     //Number of Bugs (0-4+ Bugs)
            int numHiddenLayer = Math.round((labelIndex + numClasses)/2);
            int iterations = 250;
            long seed = 6;
            int repetitions = 3;
            int numLinesToSkip = 0;
            String delimiter = ",";
            boolean bEnableFloat = true;

            //Get Data
            CSVRecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter, numClasses);
            recordReader.initialize(new FileSplit(new ClassPathResource(project).getFile())); //jdtAllMetrics, luceneAllMetrics, mylynAllMetrics
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 10000, labelIndex, numClasses);
            DataSet allData = dataSetIterator.next();
            CSVWriter writer = new CSVWriter("results_" + dateString + " " + project);

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .activation("sigmoid")
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-2)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(numHiddenLayer).build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenLayer).nOut(numClasses).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation("softmax").nIn(numClasses).nOut(numClasses).build())
                    .backprop(true)
                    .pretrain(false)
                    .build();


            float examples = 0;
            //Shuffle, Train and Evaluate k-foldIterations, repeat
            for (int k = 0; k < repetitions; k++) {
                //Initialize Model and set Score Listener
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(100));

                //Shuffle Data and initialize K-Fold-Iterator
                allData.shuffle();
                KFoldIterator kFoldIter = new KFoldIterator(allData);
                int kCount = 0;
                while (kFoldIter.hasNext()) {
                    Log.info("-----------------------------");
                    Log.info("KFold: " + kCount);
                    DataSet trainingData = kFoldIter.next();
                    DataSet testData = kFoldIter.testFold();
                    model.fit(trainingData);

                    //Evaluate the model on the test set
                    INDArray output = model.output(testData.getFeatureMatrix());
                    Integer[] bugs = new Integer[output.rows()];
                    Integer[] values = new Integer[output.rows()];
                    double[] bugsFloat = new double[output.rows()];
                    double[] valuesFloat = new double[output.rows()];
                    INDArray labels = testData.getLabels();
                    for (int i = 0; i < output.rows(); i++) {
                        float highestGuess = 0;
                        float highestValue = 0;
                        for (int j = 0; j < output.columns(); j++) {
                            float guessValue = output.getFloat(i, j);
                            float value = labels.getFloat(i, j);
                            if (guessValue > highestGuess) {
                                highestGuess = guessValue;
                                bugs[i] = j;
                            }
                            if (value > highestValue) {
                                highestValue = value;
                                values[i] = j;
                            }
                            bugsFloat[i] += guessValue*j;
                            valuesFloat[i] += value*j;
                        }
                    }
                    if (!bEnableFloat){
                        evaluator.evaluate(bugs, values, writer, examples, project);
                    }
                    else{
                        evaluator.evaluateFloat(bugsFloat, valuesFloat, writer, examples, project);
                    }
                    examples++;
                    kCount++;
                }
            }
            double rmse = evaluator.sumRmse / examples;
            float pre = evaluator.totalPrecision / examples;
            float rec = evaluator.totalRecall / examples;
            float acc = evaluator.totalAccuracy / examples;
            float accN = evaluator.totalAccuracyNumber / examples;
            Log.info("TOTAL Average RMSE: " + rmse);
            writer.writeLine(-1, "TOTAL RMSE", accN, acc, pre, rec, rmse);
            writer.close();
            evaluator.clear();
        }
    }
}