package org.ethz.las.bandit.logs;

import java.io.IOException;
import java.util.List;

public interface LogLineReader<Context, Action, Reward> {

	public LogLine<Context, Action, Reward> read() throws IOException;

	public boolean hasNext() throws IOException;

	public void close() throws IOException;

	/** @returns the possible actions corresponding to the last read log line. */
	public List<Action> getPossibleActions();
}
  