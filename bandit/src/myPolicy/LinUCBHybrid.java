package myPolicy;

import java.io.File;
import java.io.FileNotFoundException;
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
	
	private Map<Integer, DoubleMatrix> invA;
	private Map<Integer, DoubleMatrix> invA0_BT_invA;
	private Map<Integer, DoubleMatrix> invA_B_invA0_BT_invA;
	
	DoubleMatrix A0;
	DoubleMatrix b0;
	
	DoubleMatrix invA0;

	private int d = 6;
	private int k = 6;
	
	private static final double alpha = 5;
	
	// Here you can load the article features.
	public LinUCBHybrid(String articleFilePath) {
		A = new HashMap<>();
		B = new HashMap<>();
		b = new HashMap<>();
		z = new HashMap<>();
		
		A0 = DoubleMatrix.eye(k);
		invA0 = DoubleMatrix.eye(k);
		
		b0 = DoubleMatrix.zeros(k, 1);
		
		invA = new HashMap<>();
		invA0_BT_invA = new HashMap<>();
		invA_B_invA0_BT_invA = new HashMap<>();
		
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
  			counterMap.put(id, 0);
  			
  			invA.put(id, DoubleMatrix.eye(d));
  			invA0_BT_invA.put(id, DoubleMatrix.zeros(k, d));
  			invA_B_invA0_BT_invA.put(id, DoubleMatrix.zeros(d, d));
		}
		
		sc.close();
	}
  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		double maxP = 0;
  		Article maxArt = possibleActions.get(0);

  		DoubleMatrix beta = invA0.mmul(b0);

  		DoubleMatrix x = new DoubleMatrix(visitor.getFeatures());
  		DoubleMatrix xT = x.transpose();

  		for (Article a: possibleActions) {
  			int count = counterMap.get(a.getID());
  			
  			if (count <= 100 || Math.random() > .90) {
  				
  	  			DoubleMatrix Ba = B.get(a.getID());

  	  			DoubleMatrix zta = z.get(a.getID());
  	  			DoubleMatrix ztaT = zta.transpose();
  	  			DoubleMatrix invAa = invA.get(a.getID());
  				DoubleMatrix theta = invAa.mmul(b.get(a.getID()).sub(Ba.mmul(beta)));

  				double sta = ztaT.mmul(invA0).mmul(zta).get(0, 0) -
  						2 * ztaT.mmul(invA0_BT_invA.get(a.getID())).mmul(x).get(0, 0) +
  						xT.mmul(invAa).mmul(x).get(0, 0) +
  						xT.mmul(invA_B_invA0_BT_invA.get(a.getID())).mmul(x).get(0, 0);

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
  		DoubleMatrix BaT_invAa = BaT.mmul(invA.get(a.getID()));
  	
  		DoubleMatrix x = new DoubleMatrix(c.getFeatures());
  		DoubleMatrix za = z.get(a.getID());

  		/* COEFF START */
  		A0.addi(BaT_invAa.mmul(Ba));
  		b0.addi(BaT_invAa.mmul(ba));
  		
  		Aa.addi(x.mmul(x.transpose()));
  		Ba.addi(x.mmul(za.transpose()));
  		
  		DoubleMatrix invAa = Solve.pinv(Aa);
  		
  		invA.put(a.getID(), invAa);
  		/* COEFF END */
  		
  		BaT = Ba.transpose();
  		BaT_invAa = BaT.mmul(invAa);
  		
  		/* COEFF START */
  		if (reward) {
  			b0.addi(za);
  			ba.addi(x);
  		}
  	  	
  		A0.addi(za.mmul(za.transpose())).subi(BaT_invAa.mmul(Ba));
  		b0.subi(BaT_invAa.mmul(ba));
  		
  		invA0 = Solve.pinv(A0);
  		invA0_BT_invA.put(a.getID(), invA0.mmul(BaT_invAa));
		invA_B_invA0_BT_invA.put(a.getID(), invAa.mmul(Ba).mmul(invA0_BT_invA.get(a.getID())));
  		/* COEFF END */
  	}
}
