package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;

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

  public static class CommonWordsCombiner
      extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text word, Iterable<Text> counts,
        Context context) throws IOException, InterruptedException {
      int sumA = 0;
      int sumB = 0;

      for (Text val : counts) {
        String valStr = val.toString();
        switch (valStr.charAt(0)) {
          case 'a':
            sumA += Integer.valueOf(valStr.substring(1));
            break;
          case 'b':
            sumB += Integer.valueOf(valStr.substring(1));
            break;
          default:
            System.err.println("Error K-V: " + word.toString() + ": " + valStr);
        }
      }

      Text result = new Text(word.toString() + " " + Math.max(sumA, sumB));
      context.write(new Text("a"), result);
    }

  }

  public static class SortedTopCommonWordsReducer
      extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text dummyKey, Iterable<Text> results,
        Context context) throws IOException, InterruptedException {
      HashSet<String> toIgnoreSet = new HashSet<>();
      TreeMap<Integer, PriorityQueue<String>> map = new TreeMap<>((a, b) -> b - a);

      for (Text res : results) {
        String[] tokens = res.toString().split(" ", 2);
        String word = tokens[0];
        int count = Integer.valueOf(tokens[1]);

        if (toIgnoreSet.contains(word)) {
          toIgnoreSet.remove(word);
        } else {
          toIgnoreSet.add(word);
        }

        if (map.containsKey(count)) {
          PriorityQueue<String> pq = map.get(count);
          pq.add(word);
          map.put(count, pq);
        } else {
          PriorityQueue<String> pq = new PriorityQueue<>((a, b) -> b.compareTo(a));
          pq.add(word);
          map.put(count, pq);
        }
      }

      int wordsPrinted = 0;
      while (!map.isEmpty()) {
        Map.Entry<Integer, PriorityQueue<String>> entry = map.pollFirstEntry();
        Text entryCount = new Text(entry.getKey().toString());
        for (String word : entry.getValue()) {
          if (wordsPrinted == 20) { // print only top 20
            return;
          }

          if (!toIgnoreSet.contains(word)) {
            context.write(entryCount, new Text(word));
            toIgnoreSet.add(word);
            ++wordsPrinted;
          }
        }
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    GenericOptionsParser optionParser = new GenericOptionsParser(conf, args);
    String[] remainingArgs = optionParser.getRemainingArgs();
    Job job = Job.getInstance(conf, "top k common words");
    job.setJarByClass(TopkCommonWords.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(CommonWordsCombiner.class);
    job.setReducerClass(SortedTopCommonWordsReducer.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.addCacheFile(new Path(remainingArgs[2]).toUri());

    FileInputFormat.setInputPaths(job, new Path(remainingArgs[0]),
        new Path(remainingArgs[1]));
    FileOutputFormat.setOutputPath(job, new Path(remainingArgs[3]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
