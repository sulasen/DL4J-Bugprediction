package Bugprediction.tools;

import Bugprediction.BugClassificationMetrics;
import org.jfree.util.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;

import static jdk.nashorn.internal.objects.NativeMath.exp;
import static jdk.nashorn.internal.objects.NativeMath.max;

/**
 * Created by Sebi on 3/21/2017.
 */
public class Evaluator {
    private static Logger log = LoggerFactory.getLogger(BugClassificationMetrics.class);
    public float totalPrecision = 0;
    public float totalRecall = 0;
    public float totalAccuracy = 0;
    public float totalAccuracyMulti = 0;
    public float totalPRCount = 0;
    public float totalExamples = 0;
    public float totalAUC = 0;
    public float threshold = 0;
    public float sumRmse = 0;
    public float RMSEReg = 0;
    public float RMSERbC = 0;
    public float AUCCla = 0;
    public float AUCCbR = 0;

    public Evaluator(){

    }

    public void clear(){
        this.totalAccuracy = 0;
        this.totalAccuracyMulti = 0;
        this.totalPrecision = 0;
        this.totalRecall = 0;
        this.totalPRCount = 0;
        this.totalExamples = 0;
        this.totalAUC = 0;
        this.threshold = 0;
        this.sumRmse = 0;
    }

    //Calculate Accuracy, Recall, Precision, AUC etc. from Classes (as integer-array)
    public double classify(Integer[] bugs, Integer[] values, CSVWriter writer, String project) throws FileNotFoundException {
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float correctBuggy = 0;
        float predictedBuggy = 0;
        float totalBuggy = 0;
        float total = 0;
        int maxBugs = 0;
        //Max number of bugs
        for (double d:  values) {
            if (d>maxBugs){
                maxBugs = (int) d;
            }
        }
        maxBugs = maxBugs + 1; //Because zero is a valid value as well
        //True positives and false positives for auc
        int[] tp = new int[maxBugs];
        int[] fp = new int[maxBugs];

        //Evaluation
        for (int i=0; i<bugs.length;i++){
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
            if (bugs[i]==values[i]){
                correct++;
            }
            if ((values[i]!=0)){
                totalBuggy++;
            }
            if ((bugs[i]!=0)){
                predictedBuggy++;
            }
            if ((bugs[i]!=0) && (values[i]!=0)){
                correctBuggy++;
            }
            if (((bugs[i]==0) && (values[i]==0))||((bugs[i]!=0) && (values[i]!=0))){
                correctClassified++;
            }

            /* AUC Calculation */
            for (int iuc=0; iuc<maxBugs; iuc++) {
                if (bugs[i]>=iuc && (values[i]==0)){
                    fp[iuc]++;
                }
                if ((bugs[i]>=iuc) && (values[i]!=0)){
                    tp[iuc]++;
                }
            }
            total++;
        }

        /* AUC Calculation */
        float[] x = new float[maxBugs];
        float[] y = new float[maxBugs];
        float auc = 0;
        for (int iuc=0; iuc<maxBugs; iuc++) {
            x[iuc] = fp[iuc]/(total-totalBuggy);
            y[iuc] = tp[iuc]/(totalBuggy);
        }
        for (int iuc=1; iuc<maxBugs; iuc++) {
            float dx = Math.abs(x[iuc] - x[iuc-1]);
            float dy = Math.abs(y[iuc] + y[iuc-1])/2f;
            auc += dx*dy;
        }


        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);
        float accuracy = correctClassified/total;
        float accuracyNumber = correct/total;
        float precision = correctBuggy/predictedBuggy;
        float recall = correctBuggy/totalBuggy;

        //We don't want invalid values in the csv
        if (predictedBuggy>0){
            this.totalPrecision += precision;
            this.totalPRCount += 1;
        }
        else{
            precision = -1;
        }
        float f1 = 2*((precision*recall)/ (precision+recall));
        this.totalAccuracy += accuracy;
        this.totalAccuracyMulti += accuracyNumber;
        this.totalRecall += recall;
        this.totalAUC += auc;
        //this.sumRmse += curRmse;
        this.totalExamples += 1;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);
        log.info("\n\t\t\t\t\t\t\tAccuracy Number: "+accuracyNumber+
                "; \n\t\t\t\t\t\t\tAccuracy Classification: "+accuracy+
                "; \n\t\t\t\t\t\t\tPrecision: "+precision+
                "; \n\t\t\t\t\t\t\tRecall: "+recall+
                "; \n\t\t\t\t\t\t\tAUC: " +auc);

        writer.writeLine(((int) this.totalExamples), project, accuracyNumber, accuracy, precision, recall, f1, auc, 0, 0);
        return curRmse;
    }


    //Calculate Accuracy, Recall, Precision, AUC etc. for Classification by Regression
    public double classificationByRegression(double[] bugs, double[] values, CSVWriter writer, String project, float threshold) throws FileNotFoundException {
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float correctBuggy = 0;
        float predictedBuggy = 0;
        float totalBuggy = 0;
        float total = 0;
        //True positives and false positives for auc
        int nrOfT = 20;
        int[] tp = new int[nrOfT];
        int[] fp = new int[nrOfT];
        float[] thresholds = new float[nrOfT];
        for (int i=0;i<nrOfT;i++){
            thresholds[i]=(float)i/10;
        }


        //Calculate RMSE
        for (int i=0; i<bugs.length;i++) {
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);

        //Classifier (If bigger than it, then is buggy) => RMSE or 0.5 as examples
        double classifier= threshold * curRmse;

        //Evaluation
        for (int i = 0; i < bugs.length; i++) {
            if ((int) Math.floor(bugs[i] + 0.5) == values[i]) {
                correct++;
            }
            if ((values[i] != 0)) {
                totalBuggy++;
            }
            if (bugs[i] >= classifier) {
                predictedBuggy++;
            }
            if ((bugs[i] >= classifier) && (values[i] != 0)) {
                correctBuggy++;
            }
            if (((bugs[i] < classifier) && (values[i] == 0)) || ((bugs[i] >= classifier) && (values[i] != 0))) {
                correctClassified++;
            }
            total++;

            /* AUC Calculation */
            for (int iuc=0; iuc<nrOfT; iuc++) {
                double c = thresholds[iuc]*curRmse;
                if (bugs[i]>=c && (values[i]==0)){
                    fp[iuc]++;
                }
                if ((bugs[i]>=c) && (values[i]!=0)){
                    tp[iuc]++;
                }
            }
        }

        /* AUC Calculation */
        float[] x = new float[nrOfT];
        float[] y = new float[nrOfT];
        float auc = 0;
        for (int iuc=0; iuc<nrOfT; iuc++) {
            x[iuc] = fp[iuc]/(total-totalBuggy);
            y[iuc] = tp[iuc]/(totalBuggy);
        }
        for (int iuc=1; iuc<nrOfT; iuc++) {
            float dx = Math.abs(x[iuc] - x[iuc-1]);
            float dy = Math.abs(y[iuc] + y[iuc-1])/2f;
            auc += dx*dy;
        }

        float accuracy = correctClassified/total;
        float accuracyNumber = correct/total;
        float precision = correctBuggy/predictedBuggy;
        float recall = correctBuggy/totalBuggy;

        //We don't want invalid values in the csv
        if (predictedBuggy>0){
            this.totalPrecision += precision;
            this.totalPRCount += 1;
        }
        else{
            precision = -1;
        }
        float f1 = 2*((precision*recall)/ (precision+recall));
        this.totalAccuracy += accuracy;
        this.totalAUC += auc;
        this.totalAccuracyMulti += accuracyNumber;
        this.totalRecall += recall;
        //this.sumRmse += curRmse;
        this.totalExamples += 1;
        this.threshold = threshold;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);
        log.info("\n\t\t\t\t\t\t\tAccuracy Number: "+accuracyNumber+
                "; \n\t\t\t\t\t\t\tAccuracy Classification: "+accuracy+
                "; \n\t\t\t\t\t\t\tPrecision: "+precision+
                "; \n\t\t\t\t\t\t\tRecall: "+recall+
                "; \n\t\t\t\t\t\t\tAUC: " +auc);

        writer.writeLine(((int) this.totalExamples), project, accuracyNumber, accuracy, precision, recall, f1, auc, threshold, 0);
        return curRmse;
    }


    //Calculate RMSE for regression
    public double regression(double[] bugs, double[] values, CSVWriter writer, String project) throws FileNotFoundException {
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float total = 0;

        //Calculate RMSE
        for (int i=0; i<bugs.length;i++) {
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);


        this.sumRmse += curRmse;
        this.totalExamples += 1;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);

        writer.writeLine(((int) this.totalExamples), project, 0, 0, 0, 0, 0, 0, 0, curRmse);
        return curRmse;
    }


    //Calculate RMSE for regression by Classes
    public double regressionByClasses(Integer[] bugs, Integer[] values, CSVWriter writer, String project) throws FileNotFoundException {
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float total = 0;

        //Calculate RMSE
        for (int i=0; i<bugs.length;i++) {
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);


        this.sumRmse += curRmse;
        this.totalExamples += 1;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);

        writer.writeLine(((int) this.totalExamples), project, 0, 0, 0, 0, 0, 0, 0, curRmse);
        return curRmse;
    }

    public void sumUp(CSVWriter writer, CSVWriter writerSummary, String experiment){
        float rmse = this.sumRmse / this.totalExamples;
        float precision = this.totalPrecision / this.totalPRCount;
        float rec = this.totalRecall / this.totalExamples;
        float f1 = 2*((precision*rec)/ (precision+rec));
        float accuracy = this.totalAccuracy / this.totalExamples;
        float acc_multi = this.totalAccuracyMulti / this.totalExamples;
        float auc = this.totalAUC / this.totalExamples;

        Log.info("TOTAL Average RMSE: " + rmse);
        writer.writeLine(-1, "TOTAL", acc_multi, accuracy, precision, rec, f1, auc, this.threshold, rmse);
        writerSummary.storeSummary(experiment, rmse, auc);
    }
}
