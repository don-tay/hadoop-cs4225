package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.StringUtils;

public class TopkCommonWords {

  public static class TokenizerMapper
      extends Mapper<Object, Text, Text, Text> {

    private Set<String> stopwords = new HashSet<>();

    private Configuration conf;
    private BufferedReader fis;

    @Override
    public void setup(Context context) throws IOException,
        InterruptedException {
      conf = context.getConfiguration();
      URI stopwordsURI = Job.getInstance(conf).getCacheFiles()[0];
      Path stopwordsPath = new Path(stopwordsURI.getPath());
      String stopwordsFileName = stopwordsPath.getName().toString();
      parseSkipFile(stopwordsFileName);
    }

    private void parseSkipFile(String fileName) {
      try {
        fis = new BufferedReader(new FileReader(fileName));
        String stopword = null;
        while ((stopword = fis.readLine()) != null) {
          stopwords.add(stopword);
        }
      } catch (IOException ioe) {
        System.err.println("Error parsing cached file " + StringUtils.stringifyException(ioe));
      }
    }

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      Text word = new Text();
      Text val = new Text();
      HashMap<String, Integer> hash = new HashMap<>();
      String fileName = ((FileSplit) context.getInputSplit()).getPath().getName();
      String line = value.toString();
      StringTokenizer itr = new StringTokenizer(line);
      while (itr.hasMoreTokens()) {
        String tok = itr.nextToken();
        hash.merge(tok, 1, (a, b) -> a + b);
      }

      for (Map.Entry<String, Integer> elem : hash.entrySet()) {
        if (stopwords.contains(elem.getKey())) {
          continue;
        }
        String finalVal = fileName.equals("task1-input1.txt") ? "a" + elem.getValue().toString()
            : "b" + elem.getValue().toString();

        word.set(elem.getKey());
        val.set(finalVal);
        context.write(word, val);
      }
    }
  }

  public static class IntSumReducer
      extends Reducer<Text, Text, Text, Text> {
    private Text result = new Text();

    public void reduce(Text key, Iterable<Text> values,
        Context context) throws IOException, InterruptedException {
      int sumA = 0;
      int sumB = 0;

      for (Text val : values) {
        String valStr = val.toString();
        switch (valStr.charAt(0)) {
          case 'a':
            sumA += Integer.valueOf(val.toString().substring(1));
            break;
          case 'b':
            sumB += Integer.valueOf(val.toString().substring(1));
            break;
          default:
            System.err.println("Error K-V: " + key.toString() + ": " + valStr);
        }
      }

      result.set(Integer.valueOf(Math.max(sumA, sumB)).toString());
      context.write(key, result);
    }

  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
    String[] remainingArgs = optionParser.getRemainingArgs();
    Job job = Job.getInstance(conf, "top k common words");
    job.setJarByClass(TopkCommonWords.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.addCacheFile(new Path(remainingArgs[2]).toUri());

    FileInputFormat.setInputPaths(job, new Path(remainingArgs[0]),
        new Path(remainingArgs[1]));
    FileOutputFormat.setOutputPath(job, new Path(remainingArgs[3]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
