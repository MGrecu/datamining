package org.ethz.las.bandit.logs;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class RandomGenerator implements
    LogLineGenerator<Integer, Integer, Boolean> {

  private int nbOfLinesRemaining;
  private int nbOfContexts;
  private int nbOfActions;
  private Random random;
  private List<Integer> possibleActions;
  private Map<ContextAction<Integer, Integer>, Double> probas;

  public RandomGenerator(int nbOfLines, int nbOfContexts, int nbOfActions) {
    this.nbOfLinesRemaining = nbOfLines;
    this.nbOfContexts = nbOfContexts;
    this.nbOfActions = nbOfActions;
    random = new Random();
    possibleActions = new ArrayList<Integer>(nbOfActions);
    for (int i = 0; i < nbOfActions; i++)
      possibleActions.add(i);
  }

  public void setProbas(Map<ContextAction<Integer, Integer>, Double> probas) {
    this.probas = probas;
  }

  @Override
  public LogLine<Integer, Integer, Boolean> generateLogLine() {
    LogLine<Integer, Integer, Boolean> ll = new LogLine<Integer, Integer, Boolean>();
    ll.setContext(random.nextInt(nbOfContexts));
    ll.setAction(random.nextInt(nbOfActions));
    double p = 0.5;
    if (probas != null)
      p = probas.get(new ContextAction<Integer, Integer>(ll.getContext(), ll.getAction()));
    ll.setReward(random.nextDouble() < p);
    nbOfLinesRemaining--;
    return ll;
  }

  @Override
  public List<Integer> getPossibleActions() {
    return possibleActions;
  }

  @Override
  public boolean hasNext() {
    return nbOfLinesRemaining > 0;
  }

}
