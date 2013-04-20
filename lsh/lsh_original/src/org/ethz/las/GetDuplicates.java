package org.ethz.las;

import java.io.*;
import java.util.*;
import java.util.logging.Logger;
import java.util.regex.*;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

public class GetDuplicates {

  protected static final Logger log = Logger.getLogger(GetDuplicates.class.getCanonicalName());

  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> {

    Text outputKey = new Text();
    Text outputValue = new Text();

    // Used to read in the file line by line. It says that each line is in the form
    // one or more digits followed by one or more spaces followed by one or more digits
    // followed by one or more spaces followed by more than one symbol which can be
    // either lowercase letter, digit or a dot.
    Pattern p = Pattern.compile("(\\d+)\\s+(\\d+)\\s+([a-z0-9_\\.]+)");

    /**
     * Mapper reads the input and forwards it to the output setting (hash value, band) -> (document).
     */
    public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {
      Matcher m = p.matcher(value.toString());
      if (m.matches()) {
        long hashValue = Long.parseLong(m.group(1));
        int band = Integer.parseInt(m.group(2));
        String filename = m.group(3);

        outputKey.set(String.format("%d %d", hashValue, band));
        outputValue.set(filename);

        output.collect(outputKey, outputValue);
      }
    }
  }

  /**
   * One reducer takes care of one (hash value, band) pair and outputs all
   * document pairs (d_i, d_j) such that i < j. By definition, any outputed
   * pair is a candidate near duplicate pair.
   */
  public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

      // Since we only get an iterator for the values we
      // first store them in an array.
      ArrayList<String> cand = new ArrayList<String>();

      while (values.hasNext()) {
        String c = values.next().toString();
        cand.add(c);
      }

      String[] candidates = new String[cand.size()];
      candidates = cand.toArray(candidates);

      // TODO: output all pairs of near-duplicate candidates
      int n = cand.size();
      for (int i = 0; i < n; ++i) {
    	  for (int j = i + 1; j < n; ++j) {
    		  output.collect(new Text(candidates[i]), new Text(candidates[j]));
    	  }
      }
    }
  }

  public static void main(String[] args) throws Exception {

    JobConf conf = new JobConf(GenerateSignature.class);
    conf.setJobName("GetDuplicates");

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);

    conf.setMapperClass(Map.class);
    conf.setReducerClass(Reduce.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));

    JobClient.runJob(conf);
  }
}
