package org.ethz.las.bandit;

import org.ethz.las.bandit.eval.EvaluationPolicy;
import org.ethz.las.bandit.eval.Evaluator;
import org.ethz.las.bandit.eval.MyEvaluationPolicy;
import org.ethz.las.bandit.logs.FromFileLogLineGenerator;
import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.YahooLogLineReader;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;

import myPolicy.MyPolicy;

public class Main {

	private static final String DATA_FOLDER = "/Users/mircea/Documents/Master/DM/Project/Bandit/Recommender/";
	
	public static void main(String[] args) throws Exception {

		// You can hard-code the path as shown here, or you can use args to send
		// the paths to files.
		YahooLogLineReader reader = new YahooLogLineReader(DATA_FOLDER + "data/log.txt");

		FromFileLogLineGenerator<User, Article, Boolean> generator = 
				new FromFileLogLineGenerator<User, Article, Boolean>(reader);

		ContextualBanditPolicy<User, Article, Boolean> policy = new MyPolicy(DATA_FOLDER + "data/articles.txt");

		// Output log every 100 lines to standard output.
		EvaluationPolicy<User, Article, Boolean> evalPolicy = new MyEvaluationPolicy<User, Article>(
				System.out, 100, 0);

		Evaluator<User, Article, Boolean> eval = new Evaluator<User, Article, Boolean>(
				generator, evalPolicy, policy);

		System.out.println("CTR=" + eval.runEvaluation());
	}
}
