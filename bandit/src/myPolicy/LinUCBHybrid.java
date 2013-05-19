package myPolicy;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;

import org.ethz.las.bandit.logs.yahoo.Article;
import org.ethz.las.bandit.logs.yahoo.User;
import org.ethz.las.bandit.policies.ContextualBanditPolicy;
import org.ethz.las.bandit.utils.ArrayHelper;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class LinUCBHybrid implements ContextualBanditPolicy<User, Article, Boolean> {

	private Map<Integer, DoubleMatrix> A;
	private Map<Integer, DoubleMatrix> B;
	private Map<Integer, DoubleMatrix> b;
	private Map<Integer, Integer> counterMap;
	private HashMap<Integer, DoubleMatrix> z;
	
	DoubleMatrix A0;
	DoubleMatrix b0;

	private int d = 6;
	private int k = 6;
	
	private static final double alpha = 100;
	
	// Here you can load the article features.
	public LinUCBHybrid(String articleFilePath) {
		A = new HashMap<>();
		B = new HashMap<>();
		b = new HashMap<>();
		A0 = DoubleMatrix.eye(k);
		b0 = DoubleMatrix.zeros(k, 1);
		z = new HashMap<>();
		
		counterMap = new HashMap<>();

		readArticles(articleFilePath);
	}

	public void readArticles(String articleFilePath)
	{
		Scanner sc = null;
		try {
			sc = new Scanner(new File(articleFilePath));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}

		while(sc.hasNextLine()) {
			String line = sc.nextLine();
			
			String[] stringFeatures = line.split("\\s+");
			double[] doubleFeatures = ArrayHelper.stringArrayToDoubleArray(stringFeatures, 1, stringFeatures.length - 1);
			Integer id = Integer.parseInt(stringFeatures[0]);
			
			DoubleMatrix article = new DoubleMatrix(doubleFeatures);
			z.put(id, article);

  			A.put(id, DoubleMatrix.eye(d));
  			b.put(id, DoubleMatrix.zeros(d, 1));
  			B.put(id, DoubleMatrix.zeros(d, k));
  			counterMap.put(id, 1);
		}
		
		// k = z.values().iterator().next().getRows();
	}
  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		double maxP = 0;
  		Article maxArt = possibleActions.get(0);

  		DoubleMatrix invA0 = Solve.pinv(A0);
  		DoubleMatrix beta = invA0.mmul(b0);

  		DoubleMatrix x = new DoubleMatrix(visitor.getFeatures());
  		DoubleMatrix xT = x.transpose();

  		for (Article a: possibleActions) {
  			int count = counterMap.get(a.getID());
  			if (count < 10 || Math.random() > .99) {
  	  			DoubleMatrix Ba = B.get(a.getID());
  	  			DoubleMatrix BaT = Ba.transpose();

  	  			DoubleMatrix zta = z.get(a.getID());
  	  			DoubleMatrix ztaT = zta.transpose();
  	  			DoubleMatrix invA = Solve.pinv(A.get(a.getID()));
  				DoubleMatrix theta = invA.mmul(b.get(a.getID()).sub(Ba.mmul(beta)));

  				double sta = ztaT.mmul(invA0).mmul(zta).get(0, 0) -
  						2 * ztaT.mmul(invA0).mmul(BaT).mmul(invA).mmul(x).get(0, 0) +
  						xT.mmul(invA).mmul(x).get(0, 0) +
  						xT.mmul(invA).mmul(Ba).mmul(invA0).mmul(BaT).mmul(invA).mmul(x).get(0, 0);

  				double p = ztaT.mmul(beta).get(0, 0) + xT.mmul(theta).get(0, 0) + alpha * Math.sqrt(sta);

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
  		counterMap.put(a.getID(), counterMap.get(a.getID())+1);

  		DoubleMatrix Aa = A.get(a.getID());
  		DoubleMatrix Ba = B.get(a.getID());
  		DoubleMatrix BaT = Ba.transpose();

  		DoubleMatrix ba = b.get(a.getID());
  		DoubleMatrix BaTpinvAa = BaT.mmul(Solve.pinv(Aa));
  	
  		DoubleMatrix x = new DoubleMatrix(c.getFeatures());
  		DoubleMatrix za = z.get(a.getID());

  		/* COEFF START */
  		A0.addi(BaTpinvAa.mmul(Ba));
  		b0.addi(BaTpinvAa.mmul(ba));
  		
  		Aa.addi(x.mmul(x.transpose()));
  		Ba.addi(x.mmul(za.transpose()));
  		/* COEFF END */
  		
  		BaT = Ba.transpose();
  		BaTpinvAa = BaT.mmul(Solve.pinv(Aa));
  		
  		/* COEFF START */
  		if (reward) {
  			b0.addi(za);
  			ba.addi(x);
  		}
  	  	
  		A0.addi(za.mmul(za.transpose())).subi(BaTpinvAa.mmul(Ba));
  		b0.subi(BaTpinvAa.mmul(ba));
  		/* COEFF END */
  		
  	}
}
