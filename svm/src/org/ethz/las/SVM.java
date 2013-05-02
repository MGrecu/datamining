package org.ethz.las;

import java.util.*;

public class SVM {
	
  public static final int EPOCHS = 1;

  // Hyperplane weights.
  RealVector weights;

  public SVM(RealVector weights) {
    this.weights = weights;
  }

  /**
   * Instantiates an SVM from a list of training instances, for a given
   * learning rate 'eta' and regularization parameter 'lambda'.
   */
  public SVM(List<TrainingInstance> trainingSet, double eta, double lambda) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  this.weights = new RealVector(dim);
	  int t = 0;
	  
	  for (TrainingInstance ti: trainingSet) {
		  t++;
		  if (weights.dotProduct(ti.getFeatures()) * ti.getLabel() < 1) {
			  weights.add(ti.getFeatures().scale((1.0/Math.sqrt(t)) * ti.getLabel()));
			  //weights.add(ti.getFeatures().scale(eta * ti.getLabel()));
			  double factor = Math.min(1.0, 1.0 / (weights.getNorm() * lambda));
			  weights.scaleThis(factor);
		  }
	  }
  }
  
  /**
   * Simple PEGASOS, found here: http://bickson.blogspot.ch/2012/04/more-on-large-scale-svm.html
   */
  public SVM(List<TrainingInstance> trainingSet, double lambda) {
	  int dim = trainingSet.get(0).getFeatures().getDimension();
	  this.weights = new RealVector(dim);
	  int t = 0;
	  int tLast = 1;
	  double factor = 1;
	  
	  for (int i=0; i<EPOCHS; i++) {
		  for (TrainingInstance ti: trainingSet) {
			  t++;

			  if ((weights.dotProduct(ti.getFeatures()) * ti.getLabel()) < 1) {
				  factor = tLast * 1.0 / t;
				  weights = weights.scaleThis(factor);
				  weights.add(ti.getFeatures().scale((1.0/(lambda * t)) * ti.getLabel())); 
				  tLast = t;
			  }
		  }
	  }
  }

  /**
   * Instantiates SVM from weights given as a string.
   */
  public SVM(String w) {
    List<Double> ll = new LinkedList<Double>();
    Scanner sc = new Scanner(w);
    while(sc.hasNext()) {
      double coef = sc.nextDouble();
      ll.add(coef);
    }
    
    sc.close();

    double[] weights = new double[ll.size()];
    int cnt = 0;
    for (Double coef : ll)
      weights[cnt++] = coef;

    this.weights = new RealVector(weights);
  }

  /**
   * Instantiates the SVM model as the average model of the input SVMs.
   */
  public SVM(List<SVM> svmList) {
    int dim = svmList.get(0).getWeights().getDimension();
    RealVector weights = new RealVector(dim);
    for (SVM svm : svmList)
      weights.add(svm.getWeights());

    this.weights = weights.scaleThis(1.0/svmList.size());
  }

  /**
   * Given a training instance it returns the result of sign(weights'instanceFeatures).
   */
  public int classify(TrainingInstance ti) {
    @SuppressWarnings("unused")
	RealVector features = ti.getFeatures();

    double result = ti.getFeatures().dotProduct(this.weights);
    if (result >= 0) return 1;
    else return -1;
  }

  public RealVector getWeights() {
    return this.weights;
  }

  @Override
  public String toString() {
    return weights.toString();
  }
}
