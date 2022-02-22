package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.Comparator;
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
        String finalVal = elem.getValue().toString();
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
      int sum = 0;
      for (Text val : counts) {
        String valStr = val.toString();
        sum += Integer.valueOf(valStr);
      }
      Text result = new Text(word.toString() + " " + sum);
      context.write(new Text("a"), result);
    }

  }

  public static class SortedTopCommonWordsReducer
      extends Reducer<Text, Text, Text, Text> {
    public void reduce(Text dummyKey, Iterable<Text> results,
        Context context) throws IOException, InterruptedException {
      HashMap<String, Integer> largerWordCountMap = new HashMap<>();
      HashSet<String> toIgnoreSet = new HashSet<>();

      // build set to ignore words (ie. appear only once, not common in both files
      // and build map to get the larger frequency of the respective word between the
      // 2 files)
      for (Text res : results) {
        String[] tokens = res.toString().split(" ", 2);
        String word = tokens[0];
        int count = Integer.valueOf(tokens[1]);

        largerWordCountMap.merge(word, count, (a, b) -> Math.max(a, b));

        if (toIgnoreSet.contains(word)) {
          toIgnoreSet.remove(word);
        } else {
          toIgnoreSet.add(word);
        }

      }

      // build treemap to order by word frequency
      TreeMap<Integer, PriorityQueue<String>> map = new TreeMap<>((a, b) -> b - a);
      for (String word : largerWordCountMap.keySet()) {
        int count = largerWordCountMap.get(word);
        if (map.containsKey(count)) {
          PriorityQueue<String> pq = map.get(count);
          pq.add(word);
          map.put(count, pq);
        } else {
          PriorityQueue<String> pq = new PriorityQueue<>(Comparator.reverseOrder());
          pq.add(word);
          map.put(count, pq);
        }
      }

      // print words
      int wordsPrinted = 0;
      while (!map.isEmpty()) {
        Map.Entry<Integer, PriorityQueue<String>> entry = map.pollFirstEntry();
        Text entryCount = new Text(entry.getKey().toString());
        while (!(entry.getValue()).isEmpty()) {
          String word = entry.getValue().poll();
          if (wordsPrinted == 20) { // print only top 20
            return;
          }

          if (!toIgnoreSet.contains(word)) {
            context.write(entryCount, new Text(word));
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
