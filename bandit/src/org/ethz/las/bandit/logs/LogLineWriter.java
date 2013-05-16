package org.ethz.las.bandit.logs;

import java.io.IOException;

public abstract class LogLineWriter<Context, Action, Reward> {

	public abstract void write(LogLine<Context, Action, Reward> line) throws IOException;

	public abstract void close() throws IOException;
 
}
 