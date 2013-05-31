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
	private Map<Integer, Long> firstTimeMap;
	private HashMap<Integer, DoubleMatrix> z;
	
	private Map<Integer, DoubleMatrix> invA;
	private Map<Integer, DoubleMatrix> invA0_BT_invA;
	private Map<Integer, DoubleMatrix> invA_B_invA0_BT_invA;
	private HashMap<Integer, DoubleMatrix> theta;
	
	DoubleMatrix A0;
	DoubleMatrix b0;
	
	DoubleMatrix invA0;

	DoubleMatrix beta;
	
	private int d = 6;
	private int k = 6;
	
	private static final double alpha = 0.05;
	private static final double timeCoeff = 0.004;
	
	// Here you can load the article features.
	public LinUCBHybrid(String articleFilePath) {
		A = new HashMap<>();
		B = new HashMap<>();
		b = new HashMap<>();
		z = new HashMap<>();
		
		A0 = DoubleMatrix.eye(k);
		invA0 = DoubleMatrix.eye(k);
		
		b0 = DoubleMatrix.zeros(k, 1);

		beta = DoubleMatrix.zeros(k, 1);
		
		invA = new HashMap<>();
		invA0_BT_invA = new HashMap<>();
		invA_B_invA0_BT_invA = new HashMap<>();
		theta = new HashMap<>();
		
		firstTimeMap = new HashMap<>();

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
  			
  			invA.put(id, DoubleMatrix.eye(d));
  			invA0_BT_invA.put(id, DoubleMatrix.zeros(k, d));
  			invA_B_invA0_BT_invA.put(id, DoubleMatrix.zeros(d, d));
  			theta.put(id, DoubleMatrix.zeros(d, 1));
		}
		
		sc.close();
	}
  	@Override
    public Article getActionToPerform(User visitor, List<Article> possibleActions) {
  		
  		double maxP = 0;
  		Article maxArt = possibleActions.get(0);

  		DoubleMatrix x = new DoubleMatrix(visitor.getFeatures());
  		DoubleMatrix xT = x.transpose();
  		
//  		if (Math.random() <= 0.5) {
//  			int pos = (int) (Math.random() * 20);
//  			pos = pos % possibleActions.size();
//  			return possibleActions.get(pos);
//  		}

  		for (Article a: possibleActions) {
  			
  			int id = a.getID();
  			
//  			if (firstTimeMap.get(id) == null) {
//  				firstTimeMap.put(id, visitor.getTimestamp());
//  			}
//  			
//  			long timeDelta = visitor.getTimestamp() - firstTimeMap.get(id);
  			
  			DoubleMatrix zta = z.get(id);
  			DoubleMatrix ztaT = zta.transpose();
  			DoubleMatrix invAa = invA.get(id);

			double sta = ztaT.mmul(invA0).mmul(zta).get(0, 0) -
					2 * ztaT.mmul(invA0_BT_invA.get(id)).mmul(x).get(0, 0) +
					xT.mmul(invAa).mmul(x).get(0, 0) +
					xT.mmul(invA_B_invA0_BT_invA.get(id)).mmul(x).get(0, 0);

			double p = ztaT.mmul(beta).get(0, 0) + xT.mmul(theta.get(id)).get(0, 0) + alpha * Math.sqrt(sta);
			
//			p = p - timeCoeff * Math.log10(timeDelta+1);
			
			if (p >= maxP) {
				maxP = p;
				maxArt = a;
			}
  		}
  		
  		return maxArt;
  	}

  	@Override
    public void updatePolicy(User c, Article a, Boolean reward) {
  		
  		int id = a.getID();
  		
  		DoubleMatrix Aa = A.get(id);
  		DoubleMatrix Ba = B.get(id);
  		DoubleMatrix BaT = Ba.transpose();

  		DoubleMatrix ba = b.get(id);
  		DoubleMatrix BaT_invAa = BaT.mmul(invA.get(id));
  	
  		DoubleMatrix x = new DoubleMatrix(c.getFeatures());
  		DoubleMatrix za = z.get(a.getID());

  		/* COEFF START */
  		A0.addi(BaT_invAa.mmul(Ba));
  		b0.addi(BaT_invAa.mmul(ba));
  		
  		Aa.addi(x.mmul(x.transpose()));
  		Ba.addi(x.mmul(za.transpose()));
  		/* COEFF END */
  		
  		DoubleMatrix invAa = Solve.pinv(Aa);
  		invA.put(id, invAa);
  		
  		BaT = Ba.transpose();
  		BaT_invAa = BaT.mmul(invAa);
  		
  		/* COEFF START */
  		if (reward) {
  			b0.addi(za);
  			ba.addi(x);
  		}
  	  	
  		A0.addi(za.mmul(za.transpose())).subi(BaT_invAa.mmul(Ba));
  		b0.subi(BaT_invAa.mmul(ba));
  		/* COEFF END */
  		
  		invA0 = Solve.pinv(A0);
  		beta = invA0.mmul(b0);
  		
  		invA0_BT_invA.put(id, invA0.mmul(BaT_invAa));
		invA_B_invA0_BT_invA.put(id, invAa.mmul(Ba).mmul(invA0_BT_invA.get(id)));
		theta.put(id, invAa.mmul(b.get(id).sub(Ba.mmul(beta))));
  	}
}
