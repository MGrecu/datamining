package org.ethz.las.bandit.policies;

import java.util.List;

public interface ContextualBanditPolicy<Context, Action, Reward> {

  public Action getActionToPerform(Context ctx, List<Action> possibleActions);

  public void updatePolicy(Context c, Action a, Reward reward);
}
