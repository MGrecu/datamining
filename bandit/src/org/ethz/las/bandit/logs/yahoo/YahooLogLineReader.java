package org.ethz.las.bandit.logs.yahoo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.ethz.las.bandit.logs.LogLine;
import org.ethz.las.bandit.logs.LogLineReader;
import org.ethz.las.bandit.utils.ArrayHelper;


public class YahooLogLineReader implements LogLineReader<User, Article, Boolean> {

  private Scanner scan;
  private ArrayList<Article> possibleActions;

  public YahooLogLineReader(String filePath) throws FileNotFoundException {
    this.scan = new Scanner(new File(filePath));
    this.scan.useDelimiter("\n");
  }

  @Override
  public LogLine<User, Article, Boolean> read() throws IOException {
    if(! hasNext()) throw new IOException("no next line to read");

    // Get next line with line scanner.
    String line = scan.next();

    // Tokenize the line.
    String[] tokens = line.split("[\\s]+");

    // Token 0 - is timestamp, tokens 1 - 6 are user features.
    long timestamp = Long.parseLong(tokens[0]);
    double [] features = ArrayHelper.stringArrayToDoubleArray(tokens, 1, 6);

    // Token 7 is the shown article ID
    int articleId = Integer.parseInt(tokens[7]);

    // Token 8 is the click/no click information.
    boolean wasClicked = tokens[8].equals("0") ? false : true;

    // Tokens 9 to end of line are the available articles.
    int [] shownArticles = ArrayHelper.stringArrayToIntArray(tokens, 9, tokens.length - 1);

    User visitor = new User(timestamp, features);
    Article article = new Article(articleId);
    YahooLogLine logLine = new YahooLogLine(visitor, article, wasClicked);

    possibleActions = new ArrayList<Article>();
    for (int i = 0; i < shownArticles.length; ++i)
      possibleActions.add(new Article(shownArticles[i]));

    return logLine;
  }

  @Override
  public boolean hasNext() throws IOException {
    return scan.hasNext();
  }

  @Override
  public void close() throws IOException {
  }

  @Override
  public List<Article> getPossibleActions() {
    return possibleActions;
  }
}
