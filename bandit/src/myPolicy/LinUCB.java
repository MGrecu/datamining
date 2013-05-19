package myPolicy;

import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class LinUCB implements ContextualBanditPolicy<User, Article, Boolean> {

	private Map<Integer, DoubleMatrix> A;
	private Map<Integer, DoubleMatrix> b;
	private Map<Integer, Integer> counterMap;
	
	private int d = 0;
	
	private static final double alpha = 100;
	
	// Here you can load the article features.
	public LinUCB(String articleFilePath) {
		A = new HashMap<>();
		b = new HashMap<>();
		counterMap = new HashMap<>();
	}

  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		
  		double maxP = 0;
  		Article maxArt = possibleActions.get(0);
  		
  		if (d == 0) {
  			d = visitor.getFeatures().length;
  		}
  		
  		DoubleMatrix z = new DoubleMatrix(visitor.getFeatures());
  		
  		for (Article a: possibleActions) {
  			
  			if (A.get(a.getID()) == null) {
  				A.put(a.getID(), DoubleMatrix.eye(d));
  			}
  			
  			if (b.get(a.getID()) == null) {
  				b.put(a.getID(), DoubleMatrix.zeros(d, 1));
  			}
  			
  			if (counterMap.get(a.getID()) == null) {
  				counterMap.put(a.getID(), 1);
  			}
  			
  			int count = counterMap.get(a.getID());
  			counterMap.put(a.getID(), count+1);
  			
  			if (count < 1000 || count % 30 == 0) {
  				
  				DoubleMatrix invA = Solve.pinv(A.get(a.getID()));
  				DoubleMatrix theta = invA.mmul(b.get(a.getID()));
  				double firstTerm = theta.transpose().mmul(z).get(0, 0);
  				double secondTerm = alpha * Math.sqrt(z.transpose().mmul(invA).mmul(z).get(0, 0)); 
  				double p = firstTerm + secondTerm;
  				
  				if (p > maxP) {
  					maxP = p;
  					maxArt = a;
  				}
  			}
  		}
  		
  		return maxArt;
  	}

  	@Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  		
  		DoubleMatrix z = new DoubleMatrix(c.getFeatures());
  		
  		A.get(a.getID()).addi(z.mmul(z.transpose()));
  		
  		if (reward) {
  			b.get(a.getID()).addi(z);
  		}
  	}
}
