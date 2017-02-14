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

    public CSVWriter(String resultfile) throws FileNotFoundException{
        //Set up result File
        printWriter = new PrintWriter(new File(resultfile));
        StringBuilder sb = new StringBuilder();
        sb.append("run");
        sb.append(',');
        sb.append("project");
        sb.append(',');
        sb.append("acc_multi");
        sb.append(',');
        sb.append("acc_bin");
        sb.append(',');
        sb.append("precision");
        sb.append(',');
        sb.append("recall");
        sb.append(',');
        sb.append("rmse");
        sb.append('\n');
        printWriter.write(sb.toString());
    }

    public void writeLine(int run_id, String project, float accuracy_multi, float accuracy_binary, float precision, float recall, double rmse){
        StringBuilder sb = new StringBuilder();
        if (run_id>-1)
            sb.append(run_id);
        else
            sb.append("RESULTS");
        sb.append(',');
        sb.append(project);
        sb.append(',');
        sb.append(accuracy_multi);
        sb.append(',');
        sb.append(accuracy_binary);
        sb.append(',');
        sb.append(precision);
        sb.append(',');
        sb.append(recall);
        sb.append(',');
        sb.append(rmse);
        sb.append('\n');
        printWriter.write(sb.toString());
    }

    public void close(){
        printWriter.flush();
        printWriter.close();
    }
}
