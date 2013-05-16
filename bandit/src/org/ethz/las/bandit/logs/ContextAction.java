package org.ethz.las.bandit.logs;

public class ContextAction<Context, Action> {

	private Action action;
	private Context context;

	public ContextAction(Context c, Action a) {
		this.action = a;
		this.context = c;
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

	@Override
	public int hashCode() {
		return action.hashCode() * 1001 + context.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof ContextAction))
			return false;
		return ((ContextAction<?, ?>) o).getAction().equals(action)
				&& ((ContextAction<?, ?>) o).getContext().equals(context);
	}

}
