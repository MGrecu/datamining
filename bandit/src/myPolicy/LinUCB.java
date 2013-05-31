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
	private Map<Integer, Long> firstTimeMap;
	
	private Map<Integer, DoubleMatrix> invA;
	private HashMap<Integer, DoubleMatrix> theta;
	
	private int d = 6;
	
	private static final double alpha = 0.05;
	private static final double timeCoeff = 0.001;
	
	// Here you can load the article features.
	public LinUCB(String articleFilePath) {
		A = new HashMap<>();
		b = new HashMap<>();
		
		invA = new HashMap<>();
		theta = new HashMap<>();
		
		firstTimeMap = new HashMap<>();
	}

  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		
  		double maxP = 0;
  		Article maxArt = possibleActions.get(0);
  		
  		DoubleMatrix z = new DoubleMatrix(visitor.getFeatures());
  		
  		for (Article a: possibleActions) {
  			
  			int id = a.getID();
  			
//  			if (firstTimeMap.get(id) == null) {
//  				firstTimeMap.put(id, visitor.getTimestamp());
//  			}
//  			long timeDelta = visitor.getTimestamp() - firstTimeMap.get(id);
  			
  			if (A.get(id) == null) {
  				A.put(id, DoubleMatrix.eye(d));
  				invA.put(id, DoubleMatrix.eye(d));
  				b.put(id, DoubleMatrix.zeros(d, 1));
  				theta.put(id, DoubleMatrix.zeros(d, 1));
  			}
  			
			double firstTerm = theta.get(id).transpose().mmul(z).get(0, 0);
			double secondTerm = alpha * Math.sqrt(z.transpose().mmul(invA.get(id)).mmul(z).get(0, 0)); 
			double p = firstTerm + secondTerm;
			
//			p = p - timeCoeff * Math.log10(timeDelta+1);
			
			if (p > maxP) {
				maxP = p;
				maxArt = a;
			}
  		}
  		
  		return maxArt;
  	}

  	@Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  		
  		int id = a.getID();
  		
  		DoubleMatrix z = new DoubleMatrix(c.getFeatures());
  		
  		A.get(id).addi(z.mmul(z.transpose()));
  		
  		if (reward) {
  			b.get(id).addi(z);
  		}
  		
  		invA.put(id, Solve.pinv(A.get(id)));
  		theta.put(id, invA.get(id).mmul(b.get(id)));
  	}
}
