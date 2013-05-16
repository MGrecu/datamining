package org.ethz.las.bandit.eval;

import org.ethz.las.bandit.logs.LogLine;
import org.ethz.las.bandit.logs.LogLineGenerator;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;


public class Evaluator<Context, Action, Reward> {

  private LogLineGenerator<Context, Action, Reward> generator;
  private EvaluationPolicy<Context, Action, Reward> evalPolicy;
  private ContextualBanditPolicy<Context, Action, Reward> policy;
  private int linesToSkip;

  public Evaluator(LogLineGenerator<Context, Action, Reward> generator,
      EvaluationPolicy<Context, Action, Reward> eval,
      ContextualBanditPolicy<Context, Action, Reward> policy) {
    this.generator = generator;
    this.evalPolicy = eval;
    this.policy = policy;
  }

  public Evaluator(LogLineGenerator<Context, Action, Reward> generator,
                   EvaluationPolicy<Context, Action, Reward> eval,
                   ContextualBanditPolicy<Context, Action, Reward> policy,
                   int linesToSkip) {
    this(generator, eval, policy);
    this.linesToSkip = linesToSkip;
  }

  public double runEvaluation() {
    while (generator.hasNext()) {
      LogLine<Context, Action, Reward> logLine = generator.generateLogLine();
      if(linesToSkip>0) {
        linesToSkip--;
        continue;
      }
      Action a = policy.getActionToPerform(logLine.getContext(),
          generator.getPossibleActions());
      if (!generator.getPossibleActions().contains(a))
        throw new IllegalChoiceOfArticleException();
      evalPolicy.evaluate(logLine, a);
      if (a.equals(logLine.getAction()))
        policy.updatePolicy(logLine.getContext(), logLine.getAction(),
            logLine.getReward());
    }
    return evalPolicy.getResult();
  }

}
