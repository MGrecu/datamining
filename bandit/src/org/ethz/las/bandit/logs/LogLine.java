package org.ethz.las.bandit.logs;

public class LogLine<Context, Action, Reward> {
 
	private Reward reward;
	private Action action;
	private Context context;

	public LogLine() {
	} 

	public LogLine(Context context, Action action, Reward reward) {
		this.setContext(context);
		this.setAction(action);
		this.setReward(reward);
	}

	public Reward getReward() {
		return reward;
	} 

	public void setReward(Reward reward) {
		this.reward = reward;
	}

	public Action getAction() {
		return action;
	}

	public void setAction(Action action) {
		this.action = action;
	}

	public Context getContext() {
		return context;
	}

	public void setContext(Context context) {
		this.context = context;
	}

}
