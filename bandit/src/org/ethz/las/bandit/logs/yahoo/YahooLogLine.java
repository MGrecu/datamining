package org.ethz.las.bandit.logs.yahoo;

import org.ethz.las.bandit.logs.LogLine;

public class YahooLogLine extends LogLine<User, Article, Boolean> {

  public YahooLogLine(User visitor, Article article, Boolean click) {
    super(visitor, article, click);
  }

}
