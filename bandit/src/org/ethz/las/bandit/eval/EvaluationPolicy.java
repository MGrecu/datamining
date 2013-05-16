package org.ethz.las.bandit.eval;

import org.ethz.las.bandit.logs.LogLine;

public interface EvaluationPolicy<Context, Action, Reward> {

  public void evaluate(LogLine<Context, Action, Reward> logLine, Action chosenAction);

  public double getResult();

  public void log();

}
