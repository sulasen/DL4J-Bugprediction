package Bugprediction.tools;

import Bugprediction.RegressionByClassification;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;

/**
 * Created by Sebi on 3/21/2017.
 */
public class Evaluator {
    private static Logger log = LoggerFactory.getLogger(RegressionByClassification.class);
    public float totalPrecision = 0;
    public float totalRecall = 0;
    public float totalAccuracy = 0;
    public float totalAccuracyNumber = 0;
    public float sumRmse = 0;
    public Evaluator(){

    }

    public void clear(){
        this.totalAccuracy = 0;
        this.totalAccuracyNumber = 0;
        this.totalPrecision = 0;
        this.totalRecall = 0;
    }

    public double evaluate(Integer[] bugs, Integer[] values, CSVWriter writer, float examples, String project) throws FileNotFoundException {
        //Calculate RMSE, accuracy,
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float correctBuggy = 0;
        float predictedBuggy = 0;
        float totalBuggy = 0;
        float total = 0;
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
            total++;
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);
        float accuracy = correctClassified/total;
        float accuracyNumber = correct/total;
        float precision = correctBuggy/predictedBuggy;
        float recall = correctBuggy/totalBuggy;
        this.totalAccuracy += accuracy;
        this.totalAccuracyNumber += accuracyNumber;
        this.totalPrecision += precision;
        this.totalRecall += recall;
        this.sumRmse += curRmse;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);
        log.info("\n\t\t\t\t\t\t\tAccuracy Number: "+accuracyNumber+
                "; \n\t\t\t\t\t\t\tAccuracy Classification: "+accuracy+
                "; \n\t\t\t\t\t\t\tPrecision: "+precision+
                "; \n\t\t\t\t\t\t\tRecall: " +recall);

        writer.writeLine(((int) examples), project, accuracyNumber, accuracy, precision, recall, curRmse);
        return curRmse;
    }

    public double evaluateFloat(double[] bugs, double[] values, CSVWriter writer, float examples, String project) throws FileNotFoundException {
        //Calculate RMSE, accuracy,
        float squaredErrorSum=0;
        float correct = 0;
        float correctClassified = 0;
        float correctBuggy = 0;
        float predictedBuggy = 0;
        float totalBuggy = 0;
        float total = 0;
        for (int i=0; i<bugs.length;i++) {
            squaredErrorSum += Math.pow(Math.abs(bugs[i] - values[i]), 2.0);
        }
        double curRmse = Math.sqrt(squaredErrorSum/bugs.length);
        for (int i=0; i<bugs.length;i++){
            if ((int)Math.floor(bugs[i] + 0.5)==values[i]){
                correct++;
            }
            if ((values[i]!=0)){
                totalBuggy++;
            }
            if (bugs[i]>=curRmse){
                predictedBuggy++;
            }
            if ((bugs[i]>=curRmse) && (values[i]!=0)){
                correctBuggy++;
            }
            if (((bugs[i]<curRmse) && (values[i]==0))||((bugs[i]>=curRmse) && (values[i]!=0))){
                correctClassified++;
            }
            total++;
        }
        float accuracy = correctClassified/total;
        float accuracyNumber = correct/total;
        float precision = correctBuggy/predictedBuggy;
        float recall = correctBuggy/totalBuggy;
        this.totalAccuracy += accuracy;
        this.totalAccuracyNumber += accuracyNumber;
        this.totalPrecision += precision;
        this.totalRecall += recall;
        this.sumRmse += curRmse;
        log.info("Toal Cases: " + total + "; \tCorrect Exact Guesses: "+correct+"; \tCorrectly Classified: "+correctClassified+ "; \tRMSE: " + curRmse);
        log.info("\n\t\t\t\t\t\t\tAccuracy Number: "+accuracyNumber+
                "; \n\t\t\t\t\t\t\tAccuracy Classification: "+accuracy+
                "; \n\t\t\t\t\t\t\tPrecision: "+precision+
                "; \n\t\t\t\t\t\t\tRecall: " +recall);

        writer.writeLine(((int) examples), project, accuracyNumber, accuracy, precision, recall, curRmse);
        return curRmse;
    }
}
