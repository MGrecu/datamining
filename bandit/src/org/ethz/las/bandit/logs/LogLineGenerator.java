package org.ethz.las.bandit.logs;

import java.util.List;
 
public interface LogLineGenerator<Context,Action,Reward> {

	public LogLine<Context,Action,Reward> generateLogLine() ;
	public List<Action> getPossibleActions();
	public boolean hasNext();
	
}
 