package org.ethz.las.bandit.eval;

import java.io.PrintStream;

import org.ethz.las.bandit.logs.LogLine;

public class MyEvaluationPolicy<Context, Action> implements EvaluationPolicy<Context, Action, Boolean> {

  private int clicks;
  private int evaluations;
  private int lines;
  private PrintStream logger;
  private int logFrequency;
  private int linesToSkip;

  public MyEvaluationPolicy(int linesToSkip) {
    this.linesToSkip = linesToSkip;
    clicks = 0;
    evaluations = 0;
    lines = 0;
    logFrequency = -1;
  }

  public MyEvaluationPolicy(PrintStream outputStream, int logFrequency, int linesToSkip){
    clicks = 0;
    evaluations = 0;
    this.linesToSkip = linesToSkip;
    lines = 0;
    this.logFrequency = logFrequency;
    logger = outputStream;
    logger.println("lines, evaluations, clicks, score");
  }

  @Override
  public void log() {
    logger.println(lines + ", " + evaluations + ", " + clicks + ", " + getResult());
    logger.flush();
  }

  @Override
  public double getResult() {
    return ((double) clicks) / ((double) evaluations);
  }

  @Override
  public void evaluate(LogLine<Context, Action, Boolean> logLine, Action chosenAction) {
    if(linesToSkip>0) {
      linesToSkip--;
      return;
    }
    if (logLine.getAction().equals(chosenAction)) {
      evaluations++;
      if (logLine.getReward())
        clicks++;
    }
    lines++;
    if (logFrequency != -1 && lines % logFrequency == 0)
      log();
  }

}
