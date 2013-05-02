package org.ethz.las;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

import java.io.*;
import java.util.*;

public class PSGD {

  public static final int K = 20;
	
  /**
   * The Map class has to make sure that the data is shuffled to the various machines.
   */
  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, LongWritable, Text> {
    
    /**
     * Spread the data around on K different machines.
     */
    public void map(LongWritable key, Text value, OutputCollector<LongWritable, Text> output, Reporter reporter) throws IOException {
    	
    	long bucket = Math.abs(value.toString().hashCode()) % K;
    	
    	output.collect(new LongWritable(bucket), value);
    }
  }

  /**
   * Each of K reducers has to output one file containing the hyperplane.
   */
  public static class Reduce extends MapReduceBase implements Reducer<LongWritable, Text, NullWritable, Text> {
	final static double LEARNING_RATE = 0.9;
	final static double LAMBDA = 0.0001;
	final static int EPOCHS = 1;

	Text outputValue = new Text();

    /**
     * Construct a hyperplane given the subset of training examples.
     */
    public void reduce(LongWritable key, Iterator<Text> values, OutputCollector<NullWritable, Text> output, Reporter reporter) throws IOException {

      List<TrainingInstance> trainingSet = new LinkedList<TrainingInstance>();

      while (values.hasNext()) {
        String s = values.next().toString();
        TrainingInstance instance = new TrainingInstance(s);
        trainingSet.add(instance);
      }

      // SVM model = new SVM(trainingSet, LEARNING_RATE, LAMBDA);
      SVM model = new SVM(trainingSet, LAMBDA);
      //SVM model = new SVM(trainingSet, LAMBDA, 15, 10);

      /**
       * null is important here since we don't want to do additional preprocessing
       * to remove the key. The value should be the SVM model (take a look at method
       * toString in SVM.java.
       */
      outputValue.set(model.toString());
      output.collect(null, outputValue);
    }
  }

  public static void main(String[] args) throws Exception {

    JobConf conf = new JobConf(PSGD.class);

    conf.setJobName("PSGD");

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(Text.class);

    conf.setMapperClass(Map.class);
    conf.setReducerClass(Reduce.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);
    
    conf.setNumReduceTasks(K);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));

    JobClient.runJob(conf);
  }
}
