package org.ethz.las.bandit.policies;

import java.util.List;
import java.util.Random;

public class RandomPolicy<Context, Action, Reward> implements
		ContextualBanditPolicy<Context, Action, Reward> {

	private Random random;

	public RandomPolicy() {
		this.random = new Random();
	}
  
	@Override
	public Action getActionToPerform(Context ctx, List<Action> possibleActions) {
		int randomIndex = random.nextInt(possibleActions.size());
		return possibleActions.get(randomIndex);
	}

	@Override
	public void updatePolicy(Context c, Action a, Reward reward) {
	}
}
