package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {

	// Here you can load the article features.
	public MyPolicy(String articleFilePath) {
	}

  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		return possibleActions.get(0);
  	}

  	@Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  	}
}
