package Bugprediction.tools;

import org.joda.time.DateTime;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;

/**
 * Created by Sebi on 2/7/2017.
 */
public class CSVWriter {
    PrintWriter printWriter;
    public float RMSEReg;
    public float RMSERbC;
    public float AUCCbR;
    public float AUCCla;

    public CSVWriter(String resultfile) throws FileNotFoundException{
        //Set up result File
        printWriter = new PrintWriter(new File(resultfile));
        RMSEReg = 0;
        RMSERbC = 0;
        AUCCbR = 0;
        AUCCla = 0;
    }

    public void writeConfig(int classes, int HiddenNodes, int iterations, int kFold, double learningRate){
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Classes:,%s", classes));
        sb.append('\n');
        sb.append(String.format("HiddenNodes:,%s", HiddenNodes));
        sb.append('\n');
        sb.append(String.format("iterations:,%s", iterations));
        sb.append('\n');
        sb.append(String.format("kFold:,%s", kFold));
        sb.append('\n');
        sb.append(String.format("LearningRate:,%s", learningRate));
        sb.append('\n');
        String line = "run,project,acc_multi,acc_bin,precision,recall,f1,rmse,auc,threshold";
        sb.append(line);
        sb.append('\n');
        printWriter.write(sb.toString());
    }

    public void writeLine(int run_id, String project, float accuracy_multi, float accuracy_binary, float precision, float recall, float f1, float auc, float threshold, double rmse){
        String line="";
        if (run_id>-1)
            line = String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", run_id, project, accuracy_multi, accuracy_binary, precision, recall, f1, rmse, auc, threshold);
        else
            line = String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s", "RESULTS", project, accuracy_multi, accuracy_binary, precision, recall, f1, rmse, auc, threshold);
        StringBuilder sb = new StringBuilder();
        sb.append(line);
        sb.append('\n');
        printWriter.write(sb.toString());
    }


    public void setupSummary(){
        printWriter.write("Project, RMSE RbC, RMSE Reg, AUC Cla, AUC CbR, Classes\n");
    }

    public void writeSummary(String project, int numClasses){
        String line="";
        line = String.format("%s,%s,%s,%s,%s,%s", project, this.RMSERbC, this.RMSEReg, this.AUCCla, this.AUCCbR, numClasses);
        StringBuilder sb = new StringBuilder();
        sb.append(line);
        sb.append('\n');
        printWriter.write(sb.toString());
    }

    public void storeSummary(String experiment, float rmse, float auc) {
        if (experiment == "RbC") {
            this.RMSERbC = rmse;
        } else if (experiment == "Reg") {
            this.RMSEReg = rmse;
        } else if (experiment == "CbR") {
            this.AUCCbR = auc;
        } else if (experiment == "Cla") {
            this.AUCCla = auc;
        }
    }


    public void close(){
        printWriter.flush();
        printWriter.close();
    }
}
