package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;

public class MyPolicy implements ContextualBanditPolicy<User, Article, Boolean> {

	private ContextualBanditPolicy<User, Article, Boolean> policy;
	
	public MyPolicy(String articleFilePath) {
		// policy = new LinUCB(articleFilePath);
		policy = new LinUCBHybrid(articleFilePath);
	}

  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		return policy.getActionToPerform(visitor, possibleActions);
  	}

  	@Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  		policy.updatePolicy(c, a, reward);
  	}
}
