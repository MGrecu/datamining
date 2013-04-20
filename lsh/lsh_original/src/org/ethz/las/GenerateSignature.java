package org.ethz.las;

import java.io.*;
import java.util.*;
import java.util.logging.Logger;

import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapred.*;

import org.ethz.las.HashFamily;

public class GenerateSignature {

  protected static final Logger log = Logger.getLogger(GenerateSignature.class.getCanonicalName());

  /**
   * Mapper outputs the hash value for each file and hash function.
   */
  public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, LongWritable> {

    // Decide on the number of hash functions.
    final int N_HASH_FUNCTIONS = 120;

    // Documents contain shingles with index in [0, 255].
    final int N_SHINGLES = 256;

    // Prime number that will be used as MOD for all hash functions, i.e. (ax + b) % MOD.
    final int NEXT_LARGER_PRIME = 257;

    // Seed for the random hash function generation (to make sure different
    // Mappers produce the same hash functions.
    final int SEED = 1234567;

    // Generates the hash functions based on the parameters described above.
    final SimpleHashFunction[] hashFunctions = HashFamily.generateSimpleHashFamily(N_SHINGLES, NEXT_LARGER_PRIME, N_HASH_FUNCTIONS, SEED);

    // We instantiate the key here to avoid instantiating it for every line of the document.
    // LongWritable outputValue = new LongWritable();

    // Main function, implements the mapper. Gets one line as input.
    public void map(LongWritable key, Text value, OutputCollector<Text, LongWritable> output, Reporter reporter) throws IOException {

      // Gets the name of the file that is currently being mapped.
      String currentFilename = ((FileSplit)reporter.getInputSplit()).getPath().getName();

      // Get shingle value from the current line.
      int shingle = Integer.parseInt(value.toString());

      // TODO: Output (document, hash function) -> (hash value) with the goal of calculating Min-hash for this document.
      for (int i = 0; i < N_HASH_FUNCTIONS; ++i) {
    	  output.collect(new Text(String.format("%s %d", currentFilename, i)),  // key
    			  new LongWritable(hashFunctions[i].hash(shingle))); 			// value
      }
    }
  }

  /**
   * Reducer implements the minhashing.
   */
  public static class Reduce extends MapReduceBase implements Reducer<Text, LongWritable, Text, LongWritable> {

    public void reduce(Text key, Iterator<LongWritable> values, OutputCollector<Text, LongWritable> output, Reporter reporter) throws IOException {
    	// TODO: Calculate and output minhash for document and hash function.
    	LongWritable minValue = values.next();

    	while (values.hasNext()) {
    		LongWritable crt = values.next();
    		if (minValue.get() > crt.get())
    			minValue = crt;
    	}

    	output.collect(key, minValue);
    }
  }

  public static void main(String[] args) throws Exception {

    JobConf conf = new JobConf(GenerateSignature.class);
    conf.setJobName("GenerateSignature");

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(LongWritable.class);

    conf.setMapperClass(Map.class);
    conf.setReducerClass(Reduce.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(TextOutputFormat.class);

    FileInputFormat.setInputPaths(conf, new Path(args[0]));
    FileOutputFormat.setOutputPath(conf, new Path(args[1]));

    JobClient.runJob(conf);
  }
}