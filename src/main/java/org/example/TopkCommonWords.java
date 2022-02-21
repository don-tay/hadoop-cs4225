package org.example;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class TopkCommonWords {
  public static final Log log = LogFactory.getLog(TopkCommonWords.class);

  public static class TokenizerMapper
      extends Mapper<Object, Text, Text, Text> {
    private Text word = new Text();
    private Text val = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      HashMap<String, Integer> hash = new HashMap<>();
      String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String tok = itr.nextToken();
        switch (fileName) {
          case "task1-input1.txt":
          case "task1-input2.txt":
            log.debug("token: " + tok);
            hash.merge(tok, 1, (a, b) -> a + b);
            break;
          case "stopwords.txt":
            // write through
            word.set(tok);
            val.set("C");
            context.write(word, val);
            break;
          default:
            throw new IOException("Invalid file name: " + fileName);
        }
      }

      // write combined results for inputs
      if (fileName.equals("task1-input1.txt") || fileName.equals("task1-input2.txt")) {
        String prefix = fileName.equals("task1-input1.txt") ? "A" : "B";

        for (Map.Entry<String, Integer> elem : hash.entrySet()) {
          word.set(elem.getKey());
          val.set(prefix + elem.getValue().toString());
          log.debug(elem.getKey() + ": " + (prefix + elem.getValue().toString()));
          context.write(word, val);
        }
      }
    }
  }

  public static class IntSumReducer
      extends Reducer<Text, Text, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<Text> values,
        Context context) throws IOException, InterruptedException {
      boolean ignoreKey = false;
      int sumA = 0;
      int sumB = 0;
      for (Text val : values) {
        String str = val.toString();
        switch (str.charAt(0)) {
          case 'A':
            sumA += Integer.valueOf(val.toString().substring(1));
          case 'B':
            sumB += Integer.valueOf(val.toString().substring(1));
          case 'C':
            ignoreKey = true;

        }
      }

      result.set(Math.max(sumA, sumB));

      if (!ignoreKey) {
        context.write(key, result);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "top k common words");
    job.setJarByClass(TopkCommonWords.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    FileInputFormat.setInputPaths(job, new Path(args[0]),
        new Path(args[1]),
        new Path(args[2]));
    FileOutputFormat.setOutputPath(job, new Path(args[3]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
