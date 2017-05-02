package Bugprediction;

import Bugprediction.Iterators.KFoldIterator;
import Bugprediction.tools.CSVWriter;
import Bugprediction.tools.CSVRecordReader;
import Bugprediction.tools.Evaluator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jfree.util.Log;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * @author Sébastien Broggi
 */
public class RegressionByClassification {
    private static Logger log = LoggerFactory.getLogger(RegressionByClassification.class);

    public static void main(String[] args) throws  Exception {
        Evaluator evaluator = new Evaluator();
        //String[] projects = new String[] { "mylynAllMetrics.csv" };
        String[] projects = new String[] { "mylynAllMetrics.csv", "jdtAllMetrics.csv", "luceneAllMetrics.csv", "pdeAllMetrics.csv", "equinoxAllMetrics.csv" };

        for (String project: projects) {
            int labelIndex = 32;     //32 Features
            int numClasses = 5;     //Number of Bugs (0-4+ Bugs)
            int numHiddenLayer = Math.round((labelIndex + numClasses)/2);
            int iterations = 200;
            long seed = 6;
            int repetitions = 2;
            int numLinesToSkip = 0;
            String delimiter = ",";
            boolean bClassByReg = false; //Classification by Regression or vice versa
            Date date = new Date();
            SimpleDateFormat sdfDate = new SimpleDateFormat("yyyy-MM-dd_HH.mm");
            String dateString = sdfDate.format(date);

            //Get Data
            CSVRecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter, numClasses);
            recordReader.initialize(new FileSplit(new ClassPathResource(project).getFile())); //jdtAllMetrics, luceneAllMetrics, mylynAllMetrics
            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 10000, labelIndex, numClasses);
            DataSet allData = dataSetIterator.next();
            String strRegClass = "RbC";
            if (bClassByReg){
                strRegClass = "CbR";
            }
            CSVWriter writer = new CSVWriter("results_" + dateString + " " + strRegClass + " - " + project);

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .weightInit(WeightInit.XAVIER)
                    .learningRate(0.1)
                    .regularization(true).l2(1e-2)
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(labelIndex).build())
                    .layer(1, new DenseLayer.Builder().nIn(labelIndex).nOut(numHiddenLayer).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .activation("softmax").nIn(numHiddenLayer).nOut(numClasses).build())
                    .backprop(true)
                    .pretrain(false)
                    .build();


            float examples = 0;
            //Shuffle, Train and Evaluate k-foldIterations, repeat
            for (int k = 0; k < repetitions; k++) {
                //Initialize Model and set Score Listener
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(200));

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
                    if (!bClassByReg){
                        evaluator.regressionByClassification(bugs, values, writer, project);
                    }
                    else{
                        evaluator.classificationByRegression(bugsFloat, valuesFloat, writer, project);
                    }
                    examples++;
                    kCount++;
                }
            }
            double rmse = evaluator.sumRmse / evaluator.totalExamples;
            float pre = evaluator.totalPrecision / evaluator.totalPRCount;
            float rec = evaluator.totalRecall / evaluator.totalPRCount;
            float acc = evaluator.totalAccuracy / evaluator.totalExamples;
            float accN = evaluator.totalAccuracyNumber / evaluator.totalExamples;
            Log.info("TOTAL Average RMSE: " + rmse);
            writer.writeLine(-1, "TOTAL RMSE", accN, acc, pre, rec, rmse);
            writer.close();
            evaluator.clear();
        }
    }
}