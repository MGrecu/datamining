package org.ethz.las.bandit.logs;

import java.io.IOException;
import java.util.List;

public class FromFileLogLineGenerator<Context, Action, Reward> implements
		LogLineGenerator<Context, Action, Reward> {

	private LogLineReader<Context, Action, Reward> reader;

	public FromFileLogLineGenerator(
			LogLineReader<Context, Action, Reward> reader) {
		this.reader = reader;
	}
  
	@Override
	public LogLine<Context, Action, Reward> generateLogLine() {
		LogLine<Context, Action, Reward> line = null;
		try {
			line = reader.read();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return line;
	}

	@Override
	public boolean hasNext() {
		try {
			return reader.hasNext();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return false;
	}

	@Override
	public List<Action> getPossibleActions() {
		return this.reader.getPossibleActions();
	}

}
